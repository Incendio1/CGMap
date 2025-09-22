import torch.nn.functional as F
from torch_geometric.nn import TAGConv
import torch

class TAGCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = TAGConv(58, 128)
        self.conv2 = TAGConv(128, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


