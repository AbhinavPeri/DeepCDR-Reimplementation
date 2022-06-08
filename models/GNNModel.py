from torch import nn as nn
from torch.nn import functional as F
from torch_geometric.nn import BatchNorm

from util.config import ConvType, FuncActivationFunc


class GNNBlock(nn.Module):

    def __init__(self, input_features, output_features, gnn_type: ConvType, activation: FuncActivationFunc,
                 p_dropout: float, use_bn: bool):
        super().__init__()
        self.conv = gnn_type.value(input_features, output_features)
        self.activation_func = activation
        if use_bn:
            self.bn = BatchNorm(output_features)
        self.p_dropout = p_dropout
        self.use_bn = use_bn

    def forward(self, x, edge_index):
        z = self.conv(x, edge_index)
        z = self.activation_func(z)
        if self.use_bn:
            z = self.bn(z)
        z = F.dropout(z, self.p_dropout, training=self.training)
        return z.type_as(x)
