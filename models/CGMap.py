import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing


class prop_sum(MessagePassing):
    def __init__(self, args, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.args = args
        self.layers = list(args.layers)
        self.K = len(self.layers)
        self.beta = float(getattr(args, 'gpr_beta', 0.1))
        self.init_type = args.Init
        self.legacy_fixed_self = False

        hops = np.asarray(self.layers, dtype=np.float64)

        if self.init_type == 'PPR':
            hop_temp = self.beta * (1.0 - self.beta) ** np.arange(self.K, dtype=np.float64)
            hop_temp[-1] = (1.0 - self.beta) ** self.K
            temp = np.concatenate(([1.0], hop_temp))
            self.legacy_fixed_self = True

        elif self.init_type == 'PPR_raw':
            hop_temp = self.beta * (1.0 - self.beta) ** hops
            temp = np.concatenate(([self.beta], hop_temp))

        elif self.init_type == 'SignedPPR':
            A = float(getattr(args, 'gpr_A', 0.73))
            r = float(getattr(args, 'gpr_r', 0.67))
            B = float(getattr(args, 'gpr_B', 0.546))
            s = float(getattr(args, 'gpr_s', 0.99))
            positive = A * hops * np.power(r, hops - 1.0)
            long_range = B * np.power(s, hops - 1.0)
            hop_temp = positive - long_range
            temp = np.concatenate(([self.beta], hop_temp))

        elif self.init_type == 'NPPR':
            exponents = np.concatenate(([0.0], hops))
            temp = self.beta ** exponents
            denominator = np.sum(np.abs(temp))
            if denominator > 0:
                temp /= denominator

        elif self.init_type == 'Random':
            bound = np.sqrt(3.0 / (self.K + 1))
            temp = np.random.uniform(-bound, bound, self.K + 1)
            denominator = np.sum(np.abs(temp))
            if denominator > 0:
                temp /= denominator

        elif self.init_type == 'WS':
            supplied = getattr(args, 'gpr_weights', None)
            if supplied is None:
                raise ValueError('--Init WS requires --gpr_weights.')
            expected = self.K + 1
            if len(supplied) != expected:
                raise ValueError(
                    f'--gpr_weights must contain {expected} values for layers '
                    f'{self.layers}; received {len(supplied)}.'
                )
            temp = np.asarray(supplied, dtype=np.float64)

        else:
            raise ValueError(f'Unsupported GPR initialization: {self.init_type}')

        initial = torch.tensor(temp, dtype=torch.float32)
        self.temp = Parameter(initial.clone(), requires_grad=True)
        self.register_buffer('_initial_temp', initial.clone())

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []

        if self.legacy_fixed_self:
            self_channel = x
        else:
            self_channel = self.temp[0] * x

        embed_layer.append(self_channel)

        for i, layer in enumerate(self.layers):
            cache_index = layer - 1
            weight = self.temp[i + 1]
            propagated = self.propagate(
                edge_index[cache_index],
                x=x,
                norm=edge_weight[cache_index]
            )
            embed_layer.append(weight * propagated)

        return torch.stack(embed_layer, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        with torch.no_grad():
            self.temp.copy_(self._initial_temp)


class prop_weight(MessagePassing):
    """Weighted-sum alternative."""

    def __init__(self, args, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.layers = list(args.layers)
        self.alpha = float(args.alpha)
        self.weight = Parameter(
            torch.ones(len(self.layers) + 1, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x, edge_index, edge_weight):
        z = x
        embed_layer = [self.weight[0] * x]

        for i, layer in enumerate(self.layers):
            idx = layer - 1
            propagated = self.propagate(
                edge_index[idx], x=x, norm=edge_weight[idx]
            )
            h = (1.0 - self.alpha) * propagated + self.alpha * z
            embed_layer.append(self.weight[i + 1] * h)

        return torch.stack(embed_layer, dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.fill_(1.0)


class CGMap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.agg = args.agg

        in_features = int(getattr(args, 'num_features', 58))

        self.feature_transform = Linear(in_features, args.hidden)
        self.output_layer = Linear(args.hidden, 1)

        if self.agg == 'sum':
            self.propagation = prop_sum(args)
        elif self.agg == 'weighted_sum':
            self.propagation = prop_weight(args)
        else:
            raise ValueError(f'Unsupported aggregation mode: {self.agg}')

        self.feature_groups = [(0, 16), (16, 32), (32, 48), (48, 58)]

        initial_weights = getattr(args, 'initial_weights', None)
        if initial_weights is None:
            initial_weights = args.i_w

        group_init = torch.tensor(initial_weights, dtype=torch.float32)
        self.group_weights = Parameter(group_init.clone(), requires_grad=True)
        self.register_buffer('_initial_group_weights', group_init.clone())

    def reset_parameters(self):
        self.feature_transform.reset_parameters()
        self.output_layer.reset_parameters()

        if hasattr(self.propagation, 'reset_parameters'):
            self.propagation.reset_parameters()

        with torch.no_grad():
            self.group_weights.copy_(self._initial_group_weights)

    def _apply_feature_weighting(self, x):
        parts = []
        for i, (start, end) in enumerate(self.feature_groups):
            parts.append(self.group_weights[i] * x[:, start:end])
        return torch.cat(parts, dim=1)

    def forward(self, data):
        x = self._apply_feature_weighting(data.x)

        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = F.relu(self.feature_transform(x))
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        x = self.output_layer(x)

        embed = self.propagation(
            x,
            self.args.hop_edge_index,
            self.args.hop_edge_att
        )

        return torch.sum(embed, dim=1)