# import torch
# from torch_geometric.nn import GCNConv
# import torch.nn.functional as F
#
# class GCNNet(torch.nn.Module):
#     def __init__(self,args):
#         super(GCNNet, self).__init__()
#         self.args = args
#         self.conv1 = GCNConv(58, 300)
#         self.conv2 = GCNConv(300, 100)
#         self.conv3 = GCNConv(100, 1)
#
#     def forward(self, data):
#         edge_index = data.edge_index
#
#         x = F.dropout(data.x, training=self.training)
#         x = torch.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = torch.relu(self.conv2(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = self.conv3(x, edge_index)
#
#         return x
#




import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GCNNet(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()
        self.args = args
        self.num_layers = args.num_layers  # 从参数获取网络深度

        # 动态生成隐藏层维度配置
        if self.num_layers == 1:
            hidden_dims = []
        elif self.num_layers == 2:
            hidden_dims = [100]
        else:
            # 前L-2层使用300维，最后隐藏层100维
            hidden_dims = [300] * (self.num_layers - 2) + [100]

        self.convs = torch.nn.ModuleList()

        # 构建卷积层序列
        if not hidden_dims:  # 单层直接映射
            self.convs.append(GCNConv(58, 1))
        else:
            # 输入层（58维输入）
            self.convs.append(GCNConv(58, hidden_dims[0]))
            # 中间隐藏层
            for i in range(1, len(hidden_dims)):
                self.convs.append(GCNConv(hidden_dims[i - 1], hidden_dims[i]))
            # 输出层（最终输出1维）
            self.convs.append(GCNConv(hidden_dims[-1], 1))

    def forward(self, data):
        edge_index = data.edge_index

        # 输入特征随机失活
        x = F.dropout(data.x,
                      p=self.args.dropout,
                      training=self.training)

        # 前向传播过程
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # 非最后一层处理
                x = torch.relu(x)
                x = F.dropout(x,
                              p=self.args.dropout,
                              training=self.training)

        return x