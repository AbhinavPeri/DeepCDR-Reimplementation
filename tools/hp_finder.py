import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from argparse import ArgumentParser

from data.DeepCDRDataModule import DeepCDRDataModule
from models.DeepCDRLitModule import DeepCDRLitModule
from util.config import NeededFiles
from util.utils import config_from_trial


import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import optuna
from optuna.trial import Trial
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback

import joblib

pl.seed_everything(0)


def objective(trial: Trial):
    config = config_from_trial(trial)
    model = DeepCDRLitModule(config, lr=trial.suggest_loguniform('lr', 1e-4, 5e-1))
    data.batch_size = trial.suggest_int('batch size', 10, 1000, 10)
    early_stopping = EarlyStopping(monitor="Validation loss", mode='min', divergence_threshold=20, patience=4)
    logger=TensorBoardLogger(save_dir="log", name="HP Tuning")
    trainer = pl.Trainer(logger=logger, callbacks=[early_stopping, RichModelSummary(2), RichProgressBar(), PyTorchLightningPruningCallback(trial, 'Validation loss')], log_every_n_steps=1, max_epochs=30, gpus=1 if args.use_gpu else 0)
    trainer.logger.log_hyperparams(config.to_dict())
    trainer.fit(model, data)
    return trainer.callback_metrics["Validation loss"].item()


def tune(n_trials, save_path, load_path=None):
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_warmup_steps=15)
    if load_path:
        study = joblib.load(load_path)
    else: 
        study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, save_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--use_gpu", help="Tune Hyperparameters on the GPU",
                    action="store_true")
    args = parser.parse_args()
    data = DeepCDRDataModule('data', 100, batch_size=1000)
    tune(100, 'Optuna Trials/hp_tune7.pkl', load_path='Optuna Trials/hp_tune6.pkl')
    
