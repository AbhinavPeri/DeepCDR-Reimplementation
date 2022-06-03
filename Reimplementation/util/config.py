from enum import Enum
from typing import List

import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv


class ConvType(Enum):
    GCN = GCNConv
    GAT = GATConv


class FuncActivationFunc:
    ReLU = torch.relu
    Tanh = torch.tanh


class ActivationFunc(Enum):
    ReLU = nn.ReLU
    Tanh = nn.Tanh


class GNNConfig:
    def __init__(self, conv_layers: List[int], gnn_types: List[ConvType], activations: List[FuncActivationFunc],
                 use_bn: bool):
        self.conv_layers = conv_layers
        self.gnn_types = gnn_types
        self.activations = activations
        self.use_bn = use_bn


class Conv1DConfig:
    def __init__(self, n_channels: List[int], k_sizes: List[int], activations: List[ActivationFunc],
                 pool_windows: List[int], use_bn=True):
        self.n_channels = n_channels
        self.k_sizes = k_sizes
        self.activations = activations
        self.pool_windows = pool_windows
        self.use_bn = use_bn


class FCConfig:
    def __init__(self, n_neurons: List[int], activations: List[ActivationFunc], use_bn=True):
        self.n_neurons = n_neurons
        self.activations = activations
        self.use_bn = use_bn


class DeepCDRConfig:
    def __init__(self, drug_config: GNNConfig, mutation_config: Conv1DConfig, g_expr_config: FCConfig,
                 meth_config: FCConfig, regression_conv1d: Conv1DConfig):
        self.drug_config = drug_config
        self.mutation_config = mutation_config
        self.g_expr_config = g_expr_config
        self.meth_config = meth_config
        self.regression_conv1d = regression_conv1d


class NeededFiles:
    def __init__(self, drg_info, cell_ln_info, drg_features, gene_mutation, cancer_resp, gene_exp, meth):
        self.drg_info = drg_info
        self.cell_ln_info = cell_ln_info
        self.drg_features = drg_features
        self.gene_mutation = gene_mutation
        self.cancer_resp = cancer_resp
        self.gene_exp = gene_exp
        self.meth = meth
