
import torch
import torch.nn.functional as F

from torch_geometric.nn import ARMAConv


class Arma(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = ARMAConv(58, 16,
                              2, 1,
                              True, dropout=0.75)
        self.conv2 = ARMAConv(16, 1,
                              2, 1,
                              True, dropout=0.75)
        self.args = args

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
