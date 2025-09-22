import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = GATv2Conv(58, 64, heads=4)
        self.conv2 = GATv2Conv(4 * 64, 64, heads=4)
        self.conv3 = GATv2Conv(4 * 64, 1, heads=6,
                             concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x