from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import SimpleConv
from torch_geometric.nn.dense.linear import Linear


class PMLP(torch.nn.Module):
    r"""The P(ropagational)MLP model from the `"Graph Neural Networks are
    Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
    <https://arxiv.org/abs/2212.09034>`_ paper.
    :class:`PMLP` is identical to a standard MLP during training, but then
    adopts a GNN architecture during testing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): The number of layers.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        norm (bool, optional): If set to :obj:`False`, will not apply batch
            normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module
            will not learn additive biases. (default: :obj:`True`)
    """

    def __init__(self, args):
        super().__init__()

        # 从args中获取参数
        in_channels = 58
        hidden_channels = 16
        out_channels = 1  # 二分类任务，输出维度固定为1
        num_layers = 2
        dropout = 0.5
        norm = False
        bias = True

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels, self.bias))
        for _ in range(self.num_layers - 2):
            lin = Linear(hidden_channels, hidden_channels, self.bias)
            self.lins.append(lin)
        self.lins.append(Linear(hidden_channels, out_channels, self.bias))

        self.norm = None
        if norm:
            self.norm = torch.nn.BatchNorm1d(
                hidden_channels,
                affine=False,
                track_running_stats=False,
            )

        self.conv = SimpleConv(aggr='mean')

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            torch.nn.init.xavier_uniform_(lin.weight, gain=1.414)
            if self.bias:
                torch.nn.init.zeros_(lin.bias)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        # 训练时不使用边信息，验证/测试时使用
        if not self.training and edge_index is None:
            raise ValueError("During inference, edge_index must be provided.")

        for i in range(self.num_layers):
            x = x @ self.lins[i].weight.t()
            # 只有在验证/测试时且不是最后一层才使用图传播
            if not self.training and i != self.num_layers - 1:
                x = self.conv(x, edge_index)
            if self.bias:
                x = x + self.lins[i].bias
            if i != self.num_layers - 1:
                if self.norm is not None:
                    x = self.norm(x)
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
