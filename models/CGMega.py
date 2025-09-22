import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv, GCNConv, GATConv, ChebConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
from torch.nn import Dropout, MaxPool1d


class CGMega(t.nn.Module):
    def __init__(self,args):
        super(CGMega, self).__init__()
        self.drop_rate = 0.4
        self.convs = t.nn.ModuleList()
        mid_channels = 32

        # 修改 TransformerConv，移除 edge_dim 参数
        self.convs.append(
            TransformerConv(58, 32, heads=3,
                            dropout=0.1, concat=False, beta=True)
        )
        self.convs.append(
            TransformerConv(32, 32, heads=3,
                            dropout=0.1, concat=True, beta=True)
        )

        self.ln1 = LayerNorm(in_channels=mid_channels)
        self.ln2 = LayerNorm(in_channels=32 * 3)
        self.pool = MaxPool1d(2, 2)
        self.dropout = Dropout(0.4)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(int(32 * 3 / 2), 32, weight_initializer="kaiming_uniform")
        )
        self.lins.append(
            Linear(32, 1, weight_initializer="kaiming_uniform")
        )

    def forward(self, data):
        # 确保数据在正确的设备上
        x = data.x
        edge_index = data.edge_index

        # 执行边的随机丢弃（dropout_adj）操作
        edge_index, _ = dropout_adj(edge_index, p=self.drop_rate, force_undirected=True, training=self.training)

        # 计算第一层卷积和归一化
        res = x
        x = self.convs[0](x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.ln1(x)
        # print("test")
        # 第二层卷积和归一化
        edge_index, _ = dropout_adj(edge_index, p=self.drop_rate, force_undirected=True, training=self.training)
        x = self.convs[1](x, edge_index)
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        # 最大池化层和线性层
        x = t.unsqueeze(x, 1)
        x = self.pool(x)
        x = t.squeeze(x)
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)

        # 返回sigmoid输出
        # return t.sigmoid(x)
        return x