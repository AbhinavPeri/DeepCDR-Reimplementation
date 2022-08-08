import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from argparse import ArgumentParser

from data.DeepCDRDataModule import DeepCDRDataModule
from models.DeepCDRLitModule import DeepCDRLitModule
from util.config import NeededFiles
from util.utils import config_from_trial, get_tuned_config, get_config_from_param_dict

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import optuna
from optuna.study import Study
from optuna.trial import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

import joblib



def train_best_from_study(study_path):
    study = joblib.load(study_path)
    lr, batch_size, config = get_config_from_param_dict(study.best_trial.params)
    model = DeepCDRLitModule(config, lr)
    data = DeepCDRDataModule('data', 100, batch_size)
    logger=TensorBoardLogger(save_dir="log", name="Training With GMP, Linear Layers, and random split for test data")
    checkpoint_callback = ModelCheckpoint(monitor='Validation loss', save_last=True, save_top_k=5, mode='min')
    trainer = pl.Trainer(logger=logger, callbacks=[RichModelSummary(2), RichProgressBar(), checkpoint_callback], log_every_n_steps=1, max_epochs=100, gpus=2, strategy='ddp')
    trainer.logger.log_hyperparams(config.to_dict())
    trainer.fit(model, data)
    print(trainer.test(model, data))

if __name__ == '__main__':
    for i in range(5):
        pl.seed_everything(i)
        train_best_from_study('Optuna Trials/hp_tune6.pkl')
    
