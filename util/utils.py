import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch_geometric.nn

from models.GNNModel import GNNBlock
from util.config import *

from optuna.trial import Trial

GCN = ConvType.GCN
GAT = ConvType.GAT
tanh = ActivationFunc.Tanh
relu = ActivationFunc.ReLU

activation_dict = {ActivationFunc.ReLU: torch.nn.ReLU, ActivationFunc.Tanh: torch.nn.Tanh}


def normalized_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def create_1d_cnn_block(in_channels, out_channels, kernel_size, activation: ActivationFunc, pool_window, p_dropout,
                        use_bn=True):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size), activation_dict[activation](), nn.MaxPool1d(pool_window)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_channels))
    layers.append(nn.Dropout(p_dropout))
    return nn.Sequential(*layers)


def create_1d_cnn(config: Conv1DConfig):
    n_channels = config.n_channels
    k_sizes = config.k_sizes
    activations = config.activations
    pool_windows = config.pool_windows
    assert len(n_channels) == len(k_sizes) + 1 == len(activations) + 1 == len(pool_windows) + 1
    layers = []
    for i in range(len(k_sizes)):
        layers.append(
            create_1d_cnn_block(n_channels[i], n_channels[i + 1], k_sizes[i], activations[i], pool_windows[i],
                                config.p_dropout, use_bn=config.use_bn))
    return nn.Sequential(*layers)


def create_fcn(config: FCConfig):
    n_neurons = config.n_neurons
    activations = config.activations
    output = config.output
    layers = []
    for i in range(len(activations)):
        layers.append(nn.LazyLinear(n_neurons[i]))
        layers.append(activation_dict[activations[i]]())
        if config.use_bn:
            layers.append(nn.BatchNorm1d(n_neurons[i]))
        layers.append(nn.Dropout(config.p_dropout))
    if output:
        layers.append(nn.LazyLinear(output))
    return nn.Sequential(*layers)


def create_gnn(config: GNNConfig):
    conv_layers = config.conv_layers
    gnn_types = config.gnn_types
    activations = config.activations
    p_dropout = config.p_dropout
    use_bn = config.use_bn
    layers = []
    for i in range(len(activations)):
        layers.append((GNNBlock(conv_layers[i], conv_layers[i + 1], gnn_types[i], activations[i], p_dropout, use_bn),
                       'x, edge_index -> x'))
    return torch_geometric.nn.Sequential('x, edge_index', layers)

def get_random_choices(trial: Trial, name: str, categories, n):
    return [trial.suggest_categorical(name + '_' + str(i), categories) for i in range(n)]

def get_test_config(trial: Trial):
    n_layers = trial.suggest_int('n_layers', 0, 6)
    layers = get_random_choices(trial, 'layers', [20, 64, 128, 256, 512], n_layers)
    activations = get_random_choices(trial, 'activations', [tanh, relu], n_layers)
    config = FCConfig(layers, activations, output=10)
    return config

def config_from_trial(trial: Trial):
    n_drug_layers = trial.suggest_int('n_drug_layers', 2, 5)
    drug_conv_layers = get_random_choices(trial, 'drug_conv_layers', [128, 256, 512], n_drug_layers)
    drug_conv_layers.insert(0, 75)
    drug_conv_type = trial.suggest_categorical('drug_conv_type', [GCN, GAT])
    drug_conv_types = [drug_conv_type for i in range(n_drug_layers)]
    drug_activations = get_random_choices(trial, 'drug_activations', [tanh, relu], n_drug_layers)
    drug_config = GNNConfig(drug_conv_layers, drug_conv_types, drug_activations)


    mut_channels = get_random_choices(trial, 'mut_channels', [5, 10, 20], 2)
    mut_channels.insert(0, 1)
    mut_activations = get_random_choices(trial, 'mut_activations', [tanh, relu], 2)
    mutation_config = Conv1DConfig(mut_channels, [700, 5], mut_activations, [10, 10])

    n_g_expr_layers = trial.suggest_int('n_g_expr_layers', 1, 4)
    g_expr_layers = get_random_choices(trial, 'g_expr_layers', [64, 128, 256], n_g_expr_layers)
    g_expr_activations = get_random_choices(trial, 'g_expr_activations', [tanh, relu], n_g_expr_layers - 1)
    g_expr_config = FCConfig(g_expr_layers[:-1], g_expr_activations, output=g_expr_layers[-1])

    n_meth_layers = trial.suggest_int('n_meth_layers', 1, 4)
    meth_layers = get_random_choices(trial, 'meth_layers', [64, 128, 256], n_meth_layers)
    meth_activations = get_random_choices(trial, 'meth_activations', [tanh, relu], n_meth_layers - 1)
    meth_config = FCConfig(meth_layers[:-1], meth_activations, output=meth_layers[-1])


    regressor_cnn_channels = get_random_choices(trial, 'regressor_cnn_channels', [5, 10, 20], 3)
    regressor_cnn_channels.insert(0, 1)
    regressor_cnn_activations = get_random_choices(trial, 'regressor_cnn_activations', [tanh, relu], 3)
    regressor_cnn_config = Conv1DConfig(regressor_cnn_channels, [150, 5, 5], regressor_cnn_activations, [3, 3, 3])

    n_regressor_fc_layers = trial.suggest_int('n_regressor_fc_layers', 2, 5)
    regressor_fc_layers = get_random_choices(trial, 'regressor_fc_layers', [10, 50, 100], n_regressor_fc_layers)
    regressor_fc_activations = get_random_choices(trial, 'regressor_fc_activations', [tanh, relu], n_regressor_fc_layers)
    regressor_fc_config = FCConfig(regressor_fc_layers, regressor_fc_activations, output=1)
    
    config = DeepCDRConfig(drug_config, mutation_config, g_expr_config, meth_config, regressor_cnn_config, regressor_fc_config, p_dropout=0.2, use_bn=False)
    return config

def get_preset_config():
    drug_config = GNNConfig([75, 256, 256, 256], [GCN, GCN, GCN], [relu, relu, relu])
    mutation_config = Conv1DConfig([1, 5, 10], [700, 5], [tanh, relu], [10, 10])
    g_expr_config = FCConfig([256], [tanh], output=100)
    meth_config = FCConfig([256], [tanh], output=100)
    regressor_cnn_config = Conv1DConfig([1, 10, 5, 5, 5], [150, 5, 5, 5], [relu, relu, relu, relu], [2, 3, 3, 3])
    regressor_fc_config = FCConfig([], [], output=1)
    config = DeepCDRConfig(drug_config, mutation_config, g_expr_config, meth_config, regressor_cnn_config, regressor_fc_config, 0.1, False)
    return config