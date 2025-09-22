import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge

# JKNet
class JKNet(nn.Module):
    def __init__(self, args):
        super(JKNet, self).__init__()
        self.mode='max'
        self.hidden = 16
        self.num_layers = 10
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(58, 16))
        self.bns.append(nn.BatchNorm1d(16))

        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(16, 16))
            self.bns.append(nn.BatchNorm1d(16))

        self.jk = JumpingKnowledge(mode=self.mode)
        if self.mode == 'max':
            self.fc = nn.Linear(16, 1)
        elif self.mode == 'cat':
            self.fc = nn.Linear(self.num_layers * 16, 1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        layer_out = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            layer_out.append(x)

        h = self.jk(layer_out)
        h = self.fc(h)

        return h