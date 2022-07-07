import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, GATConv

from util.config import ConvType, ActivationFunc

conv_dict = {ConvType.GCN: GCNConv, ConvType.GAT: GATConv}
activation_dict = {ActivationFunc.ReLU: torch.relu, ActivationFunc.Tanh: torch.tanh}

class GNNBlock(nn.Module):

    def __init__(self, input_features, output_features, gnn_type: ConvType, activation: ActivationFunc,
                 p_dropout: float, use_bn: bool):
        super().__init__()
        self.conv = conv_dict[gnn_type](input_features, output_features)
        self.activation_func = activation_dict[activation]
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
