import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter


class prop_sum(MessagePassing):
    def __init__(self, args, **kwargs):
        super(prop_sum, self).__init__(aggr='add', **kwargs)
        self.args = args
        self.layers = args.layers
        self.beta = 0.1
        self.K = len(args.layers)
        Gamma = None

        # assert args.Init in ['PPR', 'NPPR', 'Random', 'WS']
        if args.Init == 'PPR':
            TEMP = self.beta * (1 - self.beta) ** np.arange(self.K)
            TEMP[-1] = (1 - self.beta) ** self.K
        elif args.Init == 'NPPR':
            TEMP = (self.beta) ** np.arange(self.K)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif args.Init == 'Random':
            bound = np.sqrt(3 / (self.K))
            TEMP = np.random.uniform(-bound, bound, self.K)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif args.Init == 'WS':
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP, dtype=torch.float32))

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        embed_layer.append(x)
        if(self.layers != [0]):
            for i in range(len(self.layers)):
                gamma = self.temp[i]
                h = gamma * self.propagate(edge_index[self.layers[i] - 1], x=x, norm=edge_weight[self.layers[i] - 1])
                embed_layer.append(h)

        embed_layer = torch.stack(embed_layer, dim=1)
        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        for k in range(self.args.K + 1):
            self.temp.data[k] = self.beta * (1 - self.beta) ** k
        self.temp.data[-1] = (1 - self.beta) ** self.K
        self.prop.reset_parameters()


class prop_weight(MessagePassing):
    def __init__(self, args, **kwargs):
        super(prop_weight, self).__init__(aggr='add', **kwargs)

        self.weight = torch.nn.Parameter(torch.ones(len(args.layers) + 1), requires_grad=True)
        self.layers = args.layers
        self.alpha = args.alpha
        # PPR
        TEMP = self.beta * (1 - self.beta) ** np.arange(args.K + 1)
        TEMP[-1] = (1 - self.beta) ** args.K
        self.temp = Parameter(torch.tensor(TEMP))

    def forward(self, x, edge_index, edge_weight):
        z = x
        embed_layer = []
        embed_layer.append(self.weight[0] * x)

        for i in range(len(self.layers)):
            h = (1 - self.alpha) * self.propagate(edge_index[self.layers[i] - 1], x=x, norm=edge_weight[self.layers[i] - 1]) + self.alpha * z
            embed_layer.append(self.weight[i + 1] * h)
        embed_layer = torch.stack(embed_layer, dim = 1)
        return embed_layer

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        self.weight = torch.Parameter(torch.ones(len(self.layers) + 1), requires_grad=True)


class CGMap(torch.nn.Module):
    def __init__(self, args):
        super(CGMap, self).__init__()
        self.args = args
        self.agg = args.agg
        self.feature_transform = Linear(58, self.args.hidden)
        self.output_layer = Linear(self.args.hidden, 1)
        if self.agg == 'sum':
            self.propagation = prop_sum(args)
        elif self.agg == 'weighted_sum':
            self.propagation = prop_weight(args)
        self.feature_groups = [
            (0, 16),
            (16, 32),
            (32, 48),
            (48, 58)
        ]
        if hasattr(args, 'initial_weights') and args.initial_weights is not None:
            initial_weights = torch.tensor(args.initial_weights)
        else:
            initial_weights = torch.tensor([0.05, 0.05, 0.6, 2.3])

        self.group_weights = Parameter(initial_weights, requires_grad=True)

    def reset_parameters(self):
        self.feature_transform.reset_parameters()
        self.output_layer.reset_parameters()

    def _apply_feature_weighting(self, x):
        weighted_features = []

        for i, (start_idx, end_idx) in enumerate(self.feature_groups):
            feature_slice = x[:, start_idx:end_idx]
            weighted_slice = self.group_weights[i] * feature_slice
            weighted_features.append(weighted_slice)

        return torch.cat(weighted_features, dim=1)

    def forward(self, data):
        x, _ = data.x, data.edge_index
        weighted_features = self._apply_feature_weighting(x)
        x = F.dropout(weighted_features, p=self.args.dropout, training=self.training)
        x = F.relu(self.feature_transform(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.output_layer(x)
        embed = self.propagation(x, self.args.hop_edge_index, self.args.hop_edge_att)
        x = torch.sum(embed, dim=1)
        return x