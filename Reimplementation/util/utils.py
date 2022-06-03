import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch_geometric.nn

from Reimplementation.models.GNNModel import GNNBlock
from Reimplementation.util.config import ActivationFunc, Conv1DConfig, FCConfig, GNNConfig


def normalized_adj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def create_1d_cnn_block(in_channels, out_channels, kernel_size, activation: ActivationFunc, pool_window, use_bn=True):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size), activation.value(), nn.MaxPool1d(pool_window)]
    if use_bn:
        layers.append(nn.BatchNorm1d(out_channels))
    # layers.append(nn.Dropout(0))
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
            create_1d_cnn_block(n_channels[i], n_channels[i + 1], k_sizes[i], activations[i], pool_windows[i], use_bn=config.use_bn))
    return nn.Sequential(*layers)


def create_fcn(config: FCConfig):
    n_neurons = config.n_neurons
    activations = config.activations
    layers = []
    for i in range(len(activations)):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i + 1]))
        if config.use_bn:
            layers.append(nn.BatchNorm1d(n_neurons[i + 1]))
        layers.append(activations[i].value())
        # layers.append(nn.Dropout(0))
    return nn.Sequential(*layers)


def create_gnn(config: GNNConfig):
    conv_layers = config.conv_layers
    gnn_types = config.gnn_types
    activations = config.activations
    use_bn = config.use_bn
    layers = []
    for i in range(len(activations)):
        layers.append((GNNBlock(conv_layers[i], conv_layers[i + 1], gnn_types[i], activations[i], use_bn),
                       'x, edge_index -> x'))
    return torch_geometric.nn.Sequential('x, edge_index', layers)
