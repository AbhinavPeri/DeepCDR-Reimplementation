import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from torch_geometric.data import Data, Batch

from util.config import DeepCDRConfig
from util.utils import create_1d_cnn, create_fcn, create_gnn


class DeepCDRLitModule(pl.LightningModule):
    def __init__(self, config: DeepCDRConfig, lr=0.005):
        super(DeepCDRLitModule, self).__init__()
        self.drug_gnn = create_gnn(config.drug_config)
        self.mutation_cnn1d = create_1d_cnn(config.mutation_config)
        self.mutation_cnn1d.append(nn.Flatten()).append(nn.LazyLinear(100))
        self.g_expr_fcn = create_fcn(config.g_expr_config)
        self.meth_fcn = create_fcn(config.meth_config)
        self.regression_conv1d = create_1d_cnn(config.regression_conv1d)
        self.regression_conv1d.append(nn.Flatten()).append(nn.LazyLinear(100))
        self.regression_fc = create_fcn(config.regression_fc)
        self.config = config
        self.lr = lr
        self.example_input_array = [[Batch.from_data_list(
            [Data(x=torch.randn(100, config.drug_config.conv_layers[0]),
                  edge_index=torch.randint(0, 100, (2, 101)).long())]),
            torch.randn(1, 1, 34673), torch.randn(1, 697),
            torch.randn(1, 808), torch.randn(1)]]
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2,
                                                                  verbose=True)
        return optimizer

    def forward(self, data):
        # Unpack the raw and send it through the different networks
        drug_data, mutation_data, g_expr_data, meth_data, _ = data
        x_drug = self.drug_gnn(drug_data.x, drug_data.edge_index)
        x_drug = x_drug.reshape(drug_data.num_graphs, -1)
        x_mutation = self.mutation_cnn1d(mutation_data)
        x_g_expr = self.g_expr_fcn(g_expr_data)
        x_meth = self.meth_fcn(meth_data)

        # Concatenate all the features
        x = torch.cat((x_drug, x_mutation, x_g_expr, x_meth), -1).reshape(drug_data.num_graphs, 1, -1)
        predicted_feat = self.regression_conv1d(x)
        predicted = self.regression_fc(predicted_feat)
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
