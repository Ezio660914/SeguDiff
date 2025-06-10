# -*- coding: utf-8 -*-
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torchtuples as tt
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities.types import STEP_OUTPUT
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from typing_extensions import override

from ml_utils.utils.array_tools import to_scalar, tensor_to_array
from ml_utils.utils.path_tools import try_save_pkl
from ml_utils.utils.plot_tools import cal_subplot_layout

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LossMonitor(pl.Callback):
    def __init__(self, key_name='metrics', metrics_names=('loss',), log_dir=None, use_global_step=False, val_prefix="val_"):
        """
        find metrics_names in outputs[key_name], plot curves and save to log_dir
        """
        self.key_name = key_name
        self.log_dir = log_dir
        self.metrics_names = metrics_names
        self.step_metrics = {}
        self.epoch_metrics = {}
        self.use_global_step = use_global_step
        self.val_prefix = val_prefix

    @property
    def disabled(self):
        return self.key_name is None or self.metrics_names is None or len(self.metrics_names) == 0

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.disabled:
            return
        self.reset_step()
        self.epoch_metrics = {}

    def update_step_metrics(self, outputs, batch_idx, prefix=''):
        for k in self.metrics_names:
            k = prefix + k
            if k in outputs[self.key_name]:
                v = outputs[self.key_name][k]
                if k not in self.step_metrics.keys():
                    self.step_metrics[k] = 0
                self.step_metrics[k] = (batch_idx * self.step_metrics[k] + tensor_to_array(v)) / (batch_idx + 1)

    def update_epoch_metrics(self, epoch):
        if self.step_metrics:
            for k, v in self.step_metrics.items():
                if k not in self.epoch_metrics.keys():
                    self.epoch_metrics[k] = {}
                self.epoch_metrics[k][epoch] = v

    def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        if self.disabled or self.key_name not in outputs.keys():
            return
        self.update_step_metrics(outputs, batch_idx)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.disabled or len(self.step_metrics) == 0:
            return
        self.update_epoch_metrics(trainer.global_step if self.use_global_step else trainer.current_epoch + 1)
        self.print_epoch_metrics()
        self.log_metrics(pl_module)
        if len(next(iter(self.epoch_metrics.values()))) > 1:
            self.plot_history(trainer)
        self.reset_step()

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ):
        if self.disabled or self.key_name not in outputs.keys():
            return
        self.update_step_metrics(outputs, batch_idx, self.val_prefix)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.disabled or len(self.step_metrics) == 0:
            return
        self.update_epoch_metrics(trainer.global_step if self.use_global_step else trainer.current_epoch + 1)
        self.print_epoch_metrics()
        self.log_metrics(pl_module)
        if len(next(iter(self.epoch_metrics.values()))) > 1:
            self.plot_history(trainer)
        self.reset_step()

    def reset_step(self):
        self.step_metrics = {}

    @rank_zero_only
    def print_epoch_metrics(self):
        print('\nvalidation completed')
        for k, v in self.epoch_metrics.items():
            metrics = list(v.values())
            print(f"{k}: {metrics[-1]}")

    def log_metrics(self, pl_module):
        for k, v in self.epoch_metrics.items():
            metrics = list(v.values())
            pl_module.log(k, metrics[-1], sync_dist=True)

    def plot_history(self, trainer: pl.Trainer):
        row, col, figsize = cal_subplot_layout(len(self.metrics_names), 4)
        plt.figure(figsize=figsize)
        plt.suptitle(f"Epoch {trainer.current_epoch + 1} Training Curve")
        for i, name in enumerate(self.metrics_names):
            plt.subplot(row, col, i + 1)
            plt.title(name)
            plt.xlabel('global step' if self.use_global_step else 'epoch')
            for n in [name, self.val_prefix + name]:
                if n in self.epoch_metrics:
                    epochs = list(self.epoch_metrics[n].keys())
                    metrics = list(self.epoch_metrics[n].values())
                    plt.plot(epochs, metrics, label=n, marker='.')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        if self.log_dir is not None:
            plt.savefig(os.path.join(self.log_dir, f"loss_{'_'.join(self.metrics_names)}.png"))
        plt.show()
        # plt.close()


class MetricsTool(pl.Callback):
    def __init__(self, metrics_func, key_name='y', log_dir=None, print_val_batch_metrics=False, plot_on_train_epoch_end=False, plot_on_val_epoch_end=True, use_global_step=True, enable_on_train_epoch_end=False):
        self.key_name = key_name
        self.log_dir = log_dir
        self.metrics_func = metrics_func
        self.train_step_y = []
        self.train_epoch_metrics = {}
        self.val_step_y = []
        self.val_epoch_metrics = {}
        self.print_val_batch_metrics = print_val_batch_metrics
        self.plot_on_train_epoch_end = plot_on_train_epoch_end
        self.plot_on_val_epoch_end = plot_on_val_epoch_end
        self.use_global_step = use_global_step
        self.enable_on_train_epoch_end = enable_on_train_epoch_end

    @torch.no_grad()
    def calculate_metrics(self, *args, name_prefix=''):
        """

        :param y_pred:
        :param y_true:
        :param name_prefix:
        :return: 返回包含指标名称和指标值的字典
        """
        return {
            f"{name_prefix}{metrics_name}": to_scalar(metrics_func(*args))
            for metrics_name, metrics_func in self.metrics_func.items()
        }

    @rank_zero_only
    def plot_metrics(self, title=None):
        plt.figure()
        if len(self.train_epoch_metrics) > 0:
            train_metrics_df = pd.DataFrame.from_records(list(self.train_epoch_metrics.values()), list(self.train_epoch_metrics.keys()))
            train_metrics_df.plot(ax=plt.gca(), marker='.')
        if len(self.val_epoch_metrics) > 0:
            val_metrics_df = pd.DataFrame.from_records(list(self.val_epoch_metrics.values()), list(self.val_epoch_metrics.keys()))
            val_metrics_df.plot(ax=plt.gca(), marker='.')
        plt.title(title)
        plt.grid()
        if self.log_dir is not None:
            plt.savefig(os.path.join(self.log_dir, f"metrics_{'_'.join(list(self.metrics_func.keys()))}.png"))
        plt.show()

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int
    ) -> None:
        if self.key_name not in outputs.keys():
            return
        self.train_step_y.append(outputs[self.key_name])

        m = self.calculate_metrics(*outputs[self.key_name], name_prefix='')
        pl_module.log_dict(m, prog_bar=True, sync_dist=True)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.enable_on_train_epoch_end and len(self.train_step_y) > 0:
            epoch_y = list(map(partial(torch.cat, dim=0), zip(*self.train_step_y)))
            m = self.calculate_metrics(*epoch_y, name_prefix='')
            idx = trainer.global_step if self.use_global_step else trainer.current_epoch + 1
            self.train_epoch_metrics[idx] = m
            self.print_metrics(m)
            self.train_step_y = []

            pl_module.log_dict(m, prog_bar=True, sync_dist=True)
            if self.plot_on_train_epoch_end and len(self.train_epoch_metrics) > 1:
                self.plot_metrics(f"{'global step' if self.use_global_step else 'epoch'} {idx}")

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.key_name not in outputs.keys():
            return
        self.val_step_y.append(outputs[self.key_name])
        if self.print_val_batch_metrics:
            m = self.calculate_metrics(*outputs[self.key_name], name_prefix='val_')
            self.print_metrics(m)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        idx = trainer.global_step if self.use_global_step else trainer.current_epoch + 1

        if not self.enable_on_train_epoch_end and len(self.train_step_y) > 0:
            epoch_y = list(map(partial(torch.cat, dim=0), zip(*self.train_step_y)))
            m = self.calculate_metrics(*epoch_y, name_prefix='')
            self.train_epoch_metrics[idx] = m
            self.print_metrics(m)
            self.train_step_y = []
            pl_module.log_dict(m, prog_bar=True, sync_dist=True)

        if len(self.val_step_y) > 0:
            epoch_y = list(map(partial(torch.cat, dim=0), zip(*self.val_step_y)))
            m = self.calculate_metrics(*epoch_y, name_prefix='val_')
            self.val_epoch_metrics[idx] = m
            self.print_metrics(m)
            self.val_step_y = []
            pl_module.log_dict(m, prog_bar=True, sync_dist=True)

        if self.plot_on_val_epoch_end and (len(self.val_epoch_metrics) > 1 or len(self.train_epoch_metrics) > 1):
            self.plot_metrics(f"{'global step' if self.use_global_step else 'epoch'} {idx}")

    @rank_zero_only
    def print_metrics(self, metrics_dict: dict):
        print("\nmetrics:")
        for k, v in metrics_dict.items():
            print(f"{k}:{v:.5f}")


class PredictionRecorder(pl.Callback):
    def __init__(self, keys=('preds', 'info'), log_file_path=None):
        self.keys = keys
        self.log_file_path = log_file_path
        self.step_output = []

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        batch_data = []
        for key in self.keys:
            if key in outputs.keys():
                batch_data.append(outputs[key])
        self.step_output.append(batch_data)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.log_file_path is not None:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
            try_save_pkl(self.step_output, self.log_file_path)


class NoValProgressBar(TQDMProgressBar):
    """This class disable validation tqdm to avoid glitch of validation progress bar in PyCharm"""

    @override
    def on_train_start(self, *_: Any) -> None:
        pass

    @override
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.train_progress_bar = self.init_train_tqdm()
        super().on_train_epoch_start(trainer, *_)

    @override
    def on_train_end(self, *_: Any) -> None:
        pass

    def init_validation_tqdm(self):
        pass

    @override
    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._train_progress_bar is not None:
            self.train_progress_bar.close()

    @override
    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        pass

    @override
    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        pass

    @override
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass


class ShowSamples(pl.Callback):
    def __init__(self, keys=None, n_samples=1, n_batches=1, log_dir=None, compare_channels_keys=None):
        self.keys = keys
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_dir = log_dir
        self.compare_channels_keys = compare_channels_keys
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.n_batches is not None and batch_idx >= self.n_batches:
            return
        data_dict, info = batch
        batch_data_0 = next(iter(data_dict.values()))
        batch_size, seg_len, n_channels = batch_data_0.shape
        index = np.random.permutation(batch_size)[:self.n_samples if self.n_samples >= 0 else batch_size]
        if self.keys is not None:
            samples = {}
            for k in self.keys:
                if k in outputs.keys():
                    samples[k] = outputs[k]
            if len(samples) > 0:
                row, col, figsize = cal_subplot_layout(len(samples), (7, 5))
                for n, idx in enumerate(index):
                    plt.figure(figsize=figsize)
                    for i, (k, s) in enumerate(samples.items()):
                        plt.subplot(row, col, i + 1)
                        plt.title(k)
                        plt.plot(s[idx])
                    plt.tight_layout()
                    if self.log_dir is not None:
                        plt.savefig(os.path.join(self.log_dir, f"epoch_{trainer.current_epoch + 1}_step_{trainer.global_step}_batch_{batch_idx + 1}_example_{n + 1}.png"))
                    plt.show()
        if self.compare_channels_keys is not None:
            for n, idx in enumerate(index):
                plt.figure(figsize=(8, 3 * n_channels))
                for i in range(n_channels):
                    plt.subplot(n_channels, 1, i + 1)
                    plt.title("Channel {}".format(i + 1))
                    for name in self.compare_channels_keys:
                        if name in outputs.keys():
                            plt.plot(outputs[name][idx, :, i], label=name)
                    plt.legend(loc=(1.01, 0.7))
                plt.tight_layout()
                if self.log_dir is not None:
                    plt.savefig(os.path.join(self.log_dir, f"epoch_{trainer.current_epoch + 1}_step_{trainer.global_step}_batch_{batch_idx + 1}_example_{n + 1}_channel.png"))
                plt.show()


class TorchtuplesPruningCallback(tt.callbacks.Callback):
    def __init__(self, trial: optuna.Trial, fold_id: int = None, max_epochs: int = None):
        self.trial = trial
        self.fold_id = fold_id
        self.max_epochs = max_epochs

    def get_val_loss(self):
        scores = self.model.val_metrics.scores
        epoch, val_loss = scores['loss']['epoch'][-1], scores['loss']['score'][-1]
        return epoch, val_loss

    def on_epoch_end(self):
        if isinstance(self.trial, optuna.Trial):
            epoch, current_score = self.get_val_loss()
            if self.fold_id is not None and self.max_epochs is not None:
                epoch = self.max_epochs * self.fold_id + epoch
            self.trial.report(current_score, step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")


class PyTorchLightningPruningCallbackCV(PyTorchLightningPruningCallback):
    # Define key names of `Trial.system_attrs`.
    _EPOCH_KEY = "ddp_pl:epoch"
    _INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
    _PRUNED_KEY = "ddp_pl:pruned"

    def __init__(self, trial: optuna.trial.Trial, monitor: str, fold_id):
        super().__init__(trial, monitor)
        self.fold_id = fold_id

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = trainer.max_epochs * self.fold_id + pl_module.current_epoch
        should_stop = False

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()

            # Update intermediate value in the storage.
            _trial_id = self._trial._trial_id
            _study = self._trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(self._INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
            self._trial.storage.set_trial_system_attr(
                self._trial._trial_id, self._INTERMEDIATE_VALUE, intermediate_values
            )

        # Terminate every process if any world process decides to stop.
        should_stop = trainer.strategy.broadcast(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return

        if trainer.is_global_zero:
            # Update system_attr from global zero process.
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, self._PRUNED_KEY, True)
            self._trial.storage.set_trial_system_attr(self._trial._trial_id, self._EPOCH_KEY, epoch)
