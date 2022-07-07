import os, sys

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from data.DeepCDRDataModule import DeepCDRDataModule
from models.DeepCDRLitModule import DeepCDRLitModule
from util.config import ActivationFunc, FCConfig
from util.utils import config_from_trial, get_preset_config, create_fcn, get_test_config

import torch
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data.dataloader import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.model_summary import ModelSummary
from pl_bolts.datamodules import MNISTDataModule

import optuna
from optuna.trial import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

import joblib

pl.seed_everything(0)


class LitLinear(pl.LightningModule):
    def __init__(self, config: FCConfig, lr):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Flatten(), create_fcn(config))
        self.lr = lr
        self.loss = torch.nn.CrossEntropyLoss()
        self.example_input_array = torch.randn(32, 1, 28, 28)

    def forward(self, x):
        return self.model(x)
    
    def step(self, stage, args):
        batch , _ = args
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log(stage + "_loss", loss, prog_bar=True)
        return loss

    def training_step(self, *args):
        return self.step('train', args)
    
    def validation_step(self, *args):
        return self.step('val', args)
    
    def test_step(self, *args):
        return self.step('test', args)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

class Data(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = self.val_set = self.test_set = self.dataset = None
    
    def setup(self, stage):
        x = torch.pi * 2 * torch.rand(10000, 1)
        y = torch.sin(x)
        dataset = TensorDataset(x, y)
        train, test = Subset(dataset, np.arange(9000)), Subset(dataset, np.arange(9000, None))
        self.train_set, self.val_set = random_split(train, [8000, 1000])
        self.test_set = test

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=32)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=32)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=32)


def objective(trial: Trial):
    config = get_test_config(trial)
    model = LitLinear(config, trial.suggest_loguniform('lr', 1e-4, 0.5))
    data = MNISTDataModule('.', batch_size=trial.suggest_int('batch_size', 10, 1000, step=10), num_workers=64)
    early_stopping = EarlyStopping(monitor="val_loss", mode='min', stopping_threshold=1e-5, patience=3)
    logger=TensorBoardLogger(save_dir="log", name="HP Tuning Test MNIST")
    trainer = pl.Trainer(logger=logger, callbacks=[early_stopping, RichModelSummary(2), RichProgressBar(), PyTorchLightningPruningCallback(trial, 'val_loss')], log_every_n_steps=1, max_epochs=20)
    trainer.logger.log_hyperparams(vars(config))
    trainer.fit(model, data)
    return trainer.callback_metrics["val_loss"].item()


if __name__ == '__main__':
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)
    joblib.dump(study, "study2.pkl")