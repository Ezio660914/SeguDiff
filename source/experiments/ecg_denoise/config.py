# -*- coding: utf-8 -*-
import os
import sys

from torchmetrics import functional as tmf
from functools import partial

from source.experiments.base.config import Config
from ml_utils.trainer.callbacks import MetricsTool, PredictionRecorder, LossMonitor, NoValProgressBar, ShowSamples

import lightning.pytorch as pl

from ml_utils.metrics.snr import SignalNoiseRatio
from source.trainer.callbacks import VisualizeSegmentation

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ECGDenoiseSegmentationConfig(Config):
    def __init__(self):
        super().__init__()
        # log path
        self.log_dir_name = 'SeguDiff_logs'
        self.result_save_dir_name = "results"

        # yaml config files
        self.yaml_config = os.path.join(self.yaml_dir, "denoise_segmentation.yaml")
        self.fold_config = "folds_LUDB.yaml"

        # data
        self.window_size = 768
        self.window_sep = 128
        self.rebuild_train_segment_info = False
        self.train_segment_info_save_file = f"train_seg_info_{self.window_size}_multitasks_balance.csv"
        self.val_segment_info_save_file = f"val_seg_info_{self.window_size}_multitasks.csv"
        self.used_channels = ["ii"]
        self.num_classes = 4

        self.dataset_preprocess = True
        self.dataset_prefilter = True

        # noise
        self.train_snr = [0, 1.25, 5]
        # ma, em, bw
        self.train_hybrid_weights = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        self.val_snr = [0]
        # ma, em, bw
        self.val_hybrid_weights = [[1, 1, 1]]

        self.train_noise_start = 0
        self.train_noise_end = 650000 // 2  # 650000 // 2
        self.train_noise_used_channels = [0]
        self.val_noise_start = 650000 // 2  # 650000 // 2
        self.val_noise_end = -1
        self.val_noise_used_channels = [0]

        self.train_batch_size = 64
        self.val_batch_size = 64

        self.k_folds = 5
        self.start_fold = 0
        self.single_fold = False
        # train
        self.model_checkpoint = None
        self.training = True
        self.validate = True
        self.save_torch_model = True
        self.resume_from_checkpoint = False
        self.val_check_interval = 1000
        self.limit_val_batches = None

        self.devices = [0]
        self.max_epochs = 5

    @property
    def log_dir(self):
        return os.path.join(self.project_dir, f'logs/{self.log_dir_name}')

    @property
    def log_file(self):
        return os.path.join(self.result_save_dir, "val_results.pkl")

    @property
    def result_save_dir(self):
        return os.path.join(self.log_dir, self.result_save_dir_name)

    def get_general_callbacks(self, training=True):
        return [
            # NoValProgressBar(),
            MetricsTool({
                # dice的计算不考虑背景类
                'dice': partial(tmf.dice, average='micro', ignore_index=0),
                'f1': partial(tmf.f1_score, task="multiclass", num_classes=self.num_classes, average='macro'),
                'auc': partial(tmf.auroc, task='multiclass', num_classes=self.num_classes, average='macro'),
                'acc': partial(tmf.accuracy, task='multiclass', num_classes=self.num_classes, average='macro'),
            }, 'pqrst', self.log_dir, plot_on_train_epoch_end=False, plot_on_val_epoch_end=True, use_global_step=True),
            MetricsTool({'rmse': partial(tmf.mean_squared_error, squared=False)}, 'denoise', self.log_dir, print_val_batch_metrics=not training),
            MetricsTool({'snr': SignalNoiseRatio()}, 'denoise', self.log_dir, print_val_batch_metrics=not training),
            LossMonitor("metrics", ["loss", "p_loss", "ce_loss"], self.log_dir, use_global_step=True),
            VisualizeSegmentation(['ecg', 'ecg_noisy', 'ecg_denoised'], ['labels', 'preds'], ['BG', 'P', 'QRS', 'T'], [0], 8, 1, None)
        ]

    def get_train_callbacks(self):
        return [
            *self.get_general_callbacks(training=True),
            pl.callbacks.ModelCheckpoint(
                dirpath=self.log_dir,
                every_n_train_steps=self.val_check_interval,
                save_on_train_epoch_end=True,
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=self.log_dir,
                filename="model_{epoch}_{step}_{val_snr:.4f}_{val_rmse:.4f}_{val_dice:.4f}_{val_f1:.4f}_{val_auc:.4f}_{val_acc:.4f}",
                monitor='val_snr',
                mode='max',
                verbose=True
            ),
            pl.callbacks.ModelCheckpoint(
                dirpath=self.log_dir,
                filename="model_{epoch}_{step}_{val_snr:.4f}_{val_rmse:.4f}_{val_dice:.4f}_{val_f1:.4f}_{val_auc:.4f}_{val_acc:.4f}",
                monitor='val_dice',
                mode='max',
                verbose=True
            ),
        ]

    def get_val_callbacks(self):
        return [
            *self.get_general_callbacks(training=False),
            PredictionRecorder(['ecg', 'ecg_noisy', 'ecg_denoised', 'preds', 'labels', 'info'], self.log_file),
        ]
