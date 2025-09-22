import torch
from torch_geometric.nn import ChebConv
import torch.nn.functional as F


class EMOGINet(torch.nn.Module):
    def __init__(self, args):
        super(EMOGINet, self).__init__()
        self.args = args
        self.num_layers = args.num_layers  # 从参数中获取层数

        # 动态生成各层维度配置
        if self.num_layers == 1:
            hidden_dims = []
        elif self.num_layers == 2:
            hidden_dims = [100]
        else:
            hidden_dims = [300] * (self.num_layers - 2) + [100]  # 前L-2层300维，最后隐藏层100维

        self.convs = torch.nn.ModuleList()

        # 构建卷积层序列
        if not hidden_dims:  # 单层直接映射
            self.convs.append(ChebConv(58, 1, K=2))
        else:
            # 输入层（包含初始维度58）
            self.convs.append(ChebConv(58, hidden_dims[0], K=2))
            # 中间隐藏层
            for i in range(1, len(hidden_dims)):
                self.convs.append(ChebConv(hidden_dims[i - 1], hidden_dims[i], K=2))
            # 输出层（最终输出维度1）
            self.convs.append(ChebConv(hidden_dims[-1], 1, K=2))

    def forward(self, data):
        edge_index = data.edge_index
        x = F.dropout(data.x, p=self.args.dropout, training=self.training)  # 输入特征随机失活

        # 前向传播过程
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # 非最后一层添加激活和失活
                x = torch.relu(x)
                x = F.dropout(x, p=self.args.dropout, training=self.training)

        return x