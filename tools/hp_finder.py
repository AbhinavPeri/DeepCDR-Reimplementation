import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from argparse import ArgumentParser

from data.DeepCDRDataModule import DeepCDRDataModule
from models.DeepCDRLitModule import DeepCDRLitModule
from util.config import NeededFiles
from util.utils import config_from_trial, get_preset_config

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--use_gpu", help="Tune Hyperparameters on the GPU",
                    action="store_true")
    args = parser.parse_args()
    data = DeepCDRDataModule('data/raw', 100, batch_size=1000)
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner(n_warmup_steps=15)
    study = joblib.load('hp_tune2.pkl') # optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=50)
    joblib.dump(study, 'hp_tune3.pkl')
