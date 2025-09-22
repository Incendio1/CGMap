# import torch
# import torch.nn.functional as F
# from torch.nn import Linear
# from torch_geometric.nn import ChebConv
# from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
#
# class MTGCNNet(torch.nn.Module):
#     def __init__(self, args):
#         super(MTGCNNet, self).__init__()
#         self.args = args
#         self.conv1 = ChebConv(58, 300, K=2, normalization="sym")
#         self.conv2 = ChebConv(300, 100, K=2, normalization="sym")
#         self.conv3 = ChebConv(100, 100, K=2, normalization="sym")
#         self.conv4 = ChebConv(100, 100, K=2, normalization="sym")  # Additional conv layer
#         self.conv5 = ChebConv(100, 100, K=2, normalization="sym")  # Additional conv layer
#         self.conv6 = ChebConv(100, 100, K=2, normalization="sym")  # Additional conv layer
#         self.conv7 = ChebConv(100, 100, K=2, normalization="sym")  # Additional conv layer
#         self.conv8 = ChebConv(100, 100, K=2, normalization="sym")  # Additional conv layer
#         self.conv9 = ChebConv(100, 1, K=2, normalization="sym")  # Additional conv layer
#
#         self.lin1 = Linear(58, 100)
#         self.lin2 = Linear(58, 100)
#         self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
#         self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))
#
#     def forward(self, data):
#         edge_index, _ = dropout_adj(data.edge_index, p=0.5, force_undirected=True,
#                                     num_nodes=data.x.size()[0], training=self.training)
#         pb, _ = remove_self_loops(data.edge_index)
#         pb, _ = add_self_loops(pb)
#
#         x0 = F.dropout(data.x, training=self.training)
#         x = torch.relu(self.conv1(x0, edge_index))
#         x = F.dropout(x, training=self.training)
#         x1 = torch.relu(self.conv2(x, edge_index))
#         x = x1 + torch.relu(self.lin1(x0))
#         z = x1 + torch.relu(self.lin2(x0))
#
#         # Additional convolutions
#         x = torch.relu(self.conv4(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = torch.relu(self.conv5(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = torch.relu(self.conv6(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = torch.relu(self.conv7(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = self.conv8(x, edge_index)
#
#         pos_loss = -torch.log(torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)) + 1e-15).mean()
#         neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)
#         neg_loss = -torch.log(1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
#
#         r_loss = pos_loss + neg_loss
#         return x, r_loss, self.c1, self.c2


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops

#### 2层以上代码：
# class MTGCNNet(torch.nn.Module):
#     def __init__(self, args):
#         super(MTGCNNet, self).__init__()
#         self.args = args
#         self.num_layers = args.num_layers
#
#         # 动态构建卷积层
#         self.convs = torch.nn.ModuleList()
#         self.channel_progression = [58]  # 输入特征维度
#
#         # 构建中间层（N-1层）
#         for i in range(self.num_layers - 1):
#             in_dim = self.channel_progression[-1]
#             out_dim = 300 // (2 ** i)
#             self.convs.append(ChebConv(in_dim, out_dim, K=2, normalization="sym"))
#             self.channel_progression.append(out_dim)
#
#         # 最后一层输出维度1
#         self.convs.append(ChebConv(self.channel_progression[-1], 1, K=2, normalization="sym"))
#
#         # 动态跳跃连接配置
#         self.skip_layer_idx = self.num_layers - 2  # 总是选择最后一层中间层
#         skip_out_dim = self.channel_progression[self.skip_layer_idx + 1]
#         self.lin1 = Linear(58, skip_out_dim)
#         self.lin2 = Linear(58, skip_out_dim)
#
#         self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
#         self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))
#
#     def forward(self, data):
#         edge_index, _ = dropout_adj(data.edge_index, p=0.5,
#                                     force_undirected=True,
#                                     num_nodes=data.x.size()[0],
#                                     training=self.training)
#         E = data.edge_index
#         pb, _ = remove_self_loops(data.edge_index)
#         pb, _ = add_self_loops(pb)
#
#         x0 = F.dropout(data.x, training=self.training)
#         x = x0
#
#         # 前向传播各中间层
#         layer_outputs = []
#         for conv in self.convs[:-1]:
#             x = torch.relu(conv(x, edge_index))
#             x = F.dropout(x, training=self.training)
#             layer_outputs.append(x)
#
#         # 动态选择最后一层中间层的输出
#         x1 = layer_outputs[self.skip_layer_idx]
#
#         # 执行维度匹配的跳跃连接
#         x_skip = torch.relu(self.lin1(x0))
#         z_skip = torch.relu(self.lin2(x0))
#
#         # 最终特征融合
#         x = x1 + x_skip
#         z = x1 + z_skip
#
#         # 负采样损失计算
#         pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()
#         neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)
#         neg_loss = -torch.log(
#             1 - torch.sigmoid((z[neg_edge_index[0]] *
#
#
#
#
#
#
#                                z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
#         r_loss = pos_loss + neg_loss
#
#         # 最终预测层（保持x为最后一层中间层输出）
#         x = self.convs[-1](x, edge_index)
#         return x, r_loss, self.c1, self.c2



########单层MTGCN
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops


class MTGCNNet(torch.nn.Module):
    def __init__(self, args):
        super(MTGCNNet, self).__init__()
        self.args = args
        self.num_layers = args.num_layers

        # 动态构建卷积层
        self.convs = torch.nn.ModuleList()
        if self.num_layers == 1:
            # 单层直接映射
            self.convs.append(ChebConv(58, 1, K=2, normalization="sym"))
            lin_out_dim = 1
        else:
            # 多层结构（保持原始论文层级特征）
            hidden_dims = [300] * (self.num_layers - 2) + [100]
            # 添加各卷积层
            self.convs.append(ChebConv(58, hidden_dims[0], K=2, normalization="sym"))
            for i in range(1, len(hidden_dims)):
                self.convs.append(ChebConv(hidden_dims[i - 1], hidden_dims[i], K=2, normalization="sym"))
            self.convs.append(ChebConv(hidden_dims[-1], 1, K=2, normalization="sym"))
            lin_out_dim = hidden_dims[-1]

        # 自适应调整线性层维度
        self.lin1 = Linear(58, lin_out_dim)
        self.lin2 = Linear(58, lin_out_dim)

        # 保持原始参数配置
        self.c1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.c2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, data):
        # 边处理（保持原始逻辑）
        edge_index, _ = dropout_adj(data.edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.x.size()[0],
                                    training=self.training)
        E = data.edge_index
        pb, _ = remove_self_loops(data.edge_index)
        pb, _ = add_self_loops(pb)

        # 特征预处理
        x0 = F.dropout(data.x, p=self.args.dropout,
                       training=self.training)

        # 动态卷积层前向传播
        x = x0
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # 非最后一层处理
                x = torch.relu(x)
                x = F.dropout(x, p=self.args.dropout,
                              training=self.training)
        x1 = x  # 最终特征表示

        # 保持原始融合逻辑
        x = x1 + torch.relu(self.lin1(x0))
        z = x1 + torch.relu(self.lin2(x0))

        # 保持原始损失计算
        pos_loss = -torch.log(torch.sigmoid((z[E[0]] * z[E[1]]).sum(dim=1)) + 1e-15).mean()
        neg_edge_index = negative_sampling(pb, data.num_nodes, data.num_edges)
        neg_loss = -torch.log(
            1 - torch.sigmoid((z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)) + 1e-15).mean()
        r_loss = pos_loss + neg_loss

        return x1, r_loss, self.c1, self.c2  # 直接返回最后一层输出作为预测