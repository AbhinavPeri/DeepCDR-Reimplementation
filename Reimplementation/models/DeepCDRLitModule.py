import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from torch_geometric.data import Data, Batch

from Reimplementation.util.config import DeepCDRConfig
from Reimplementation.util.utils import create_1d_cnn, create_fcn, create_gnn


class DeepCDRLitModule(pl.LightningModule):
    def __init__(self, config: DeepCDRConfig, batch_size=32):
        super(DeepCDRLitModule, self).__init__()
        self.drug_gnn = create_gnn(config.drug_config)
        self.mutation_cnn1d = create_1d_cnn(config.mutation_config)
        self.g_expr_fcn = create_fcn(config.g_expr_config)
        self.meth_fcn = create_fcn(config.meth_config)
        self.regression_conv1d = create_1d_cnn(config.regression_conv1d)
        self.config = config
        self.batch_size = batch_size
        self.example_input_array = [[Batch.from_data_list(
            [Data(x=torch.randn(100, config.drug_config.conv_layers[0]),
                  edge_index=torch.randint(0, 100, (2, 101)).long())]),
            torch.randn(1, 1, 34673), torch.randn(1, config.g_expr_config.n_neurons[0]),
            torch.randn(1, config.meth_config.n_neurons[0]), torch.randn(1)]]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)

    def forward(self, data):
        # Unpack the data and send it through the different networks
        drug_data, mutation_data, g_expr_data, meth_data, _ = data
        x_drug = self.drug_gnn(drug_data.x, drug_data.edge_index)
        x_drug = x_drug.reshape(drug_data.num_graphs, -1)
        x_mutation = nn.Flatten()(self.mutation_cnn1d(mutation_data))
        x_mutation = nn.ReLU()(nn.Linear(x_mutation.shape[-1], 100)(x_mutation))
        x_g_expr = self.g_expr_fcn(g_expr_data)
        x_meth = self.meth_fcn(meth_data)

        # Concatenate all the features
        x = torch.cat((x_drug, x_mutation, x_g_expr, x_meth), -1).reshape(drug_data.num_graphs, 1, -1)

        # Regress the IC50 scores

        x = self.regression_conv1d(x)  # <- This line runs the final 1D convolution on the concatenated features
        x = nn.Flatten()(x)

        # These lines are the linear layer that regress the IC50 score
        # predicted = nn.Linear(x.shape[-1], 50)(x)
        # predicted = nn.ReLU()(predicted)
        predicted = nn.Linear(x.shape[-1], 1)(x)
        return predicted.squeeze()

    def evaluate(self, batch, stage: str):
        drug_data, _, _, _, ic50_scores = batch
        prediction = self.forward(batch)
        loss = F.mse_loss(prediction, ic50_scores)
        self.log(stage + " loss", loss, prog_bar=True, batch_size=drug_data.num_graphs)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate(batch, "Train")

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "Validation")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "Test")
