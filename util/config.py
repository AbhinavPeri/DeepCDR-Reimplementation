from enum import IntEnum
from typing import List

import torch
from torch import nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv


class ConvType(IntEnum):
    GCN = 0
    GAT = 1

class ActivationFunc(IntEnum):
    ReLU = 0
    Tanh = 1


class GNNConfig:
    def __init__(self, conv_layers: List[int], gnn_types: List[ConvType], activations: List[ActivationFunc],
                 p_dropout=0, use_bn=False):
        self.conv_layers = conv_layers
        self.gnn_types = gnn_types
        self.activations = activations
        self.p_dropout = p_dropout
        self.use_bn = use_bn


class Conv1DConfig:
    def __init__(self, n_channels: List[int], k_sizes: List[int], activations: List[ActivationFunc],
                 pool_windows: List[int], p_dropout=0, use_bn=False):
        self.n_channels = n_channels
        self.k_sizes = k_sizes
        self.activations = activations
        self.pool_windows = pool_windows
        self.p_dropout = p_dropout
        self.use_bn = use_bn


class FCConfig:
    def __init__(self, n_neurons: List[int], activations: List[ActivationFunc], output=None, p_dropout=0.0, use_bn=False):
        assert len(activations) == len(n_neurons)
        self.n_neurons = n_neurons
        self.activations = activations
        self.output = output
        self.p_dropout = p_dropout
        self.use_bn = use_bn


class DeepCDRConfig:
    def __init__(self, drug_config: GNNConfig, mutation_config: Conv1DConfig, g_expr_config: FCConfig,
                 meth_config: FCConfig, regression_conv1d: Conv1DConfig, regression_fc: FCConfig, p_dropout: float,
                 use_bn: bool):
        drug_config.p_dropout = p_dropout
        drug_config.use_bn = use_bn
        mutation_config.p_dropout = p_dropout
        mutation_config.use_bn = use_bn
        g_expr_config.p_dropout = p_dropout
        g_expr_config.use_bn = use_bn
        meth_config.p_dropout = p_dropout
        meth_config.use_bn = use_bn
        regression_conv1d.p_dropout = p_dropout
        regression_conv1d.use_bn = use_bn
        regression_fc.p_dropout = p_dropout
        regression_fc.use_bn = use_bn
        self.drug_config = drug_config
        self.mutation_config = mutation_config
        self.g_expr_config = g_expr_config
        self.meth_config = meth_config
        self.regression_conv1d = regression_conv1d
        self.regression_fc = regression_fc
    
    def to_dict(self):
        result = {}
        for variable, value in vars(self).items():
            for sub_variable, sub_value in vars(value).items():
                result[variable + "_" + sub_variable] = sub_value
        return result

class NeededFiles:
    def __init__(self, data_dir: str):
        self.drg_info = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv' % data_dir
        self.cell_ln_info = '%s/CCLE/Cell_lines_annotations_20181226.txt' % data_dir
        self.drg_features = '%s/GDSC/drug_graph_feat' % data_dir
        self.gene_mutation = '%s/CCLE/genomic_mutation_34673_demap_features.csv' % data_dir
        self.cancer_resp = '%s/CCLE/GDSC_IC50.csv' % data_dir
        self.gene_exp = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv' % data_dir
        self.meth = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv' % data_dir
