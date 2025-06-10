# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict

import lightning.pytorch as pl
import torch
from lightning.fabric.accelerators import find_usable_cuda_devices

from ml_utils.experiments.base.data_pipeline import IDataPreparation
from ml_utils.experiments.base.model_pipeline import IModelTrainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LightningModelTrainer(IModelTrainer):
    def __init__(self, config=None):
        super().__init__(config)

    def create_trainer(self, training=True):
        if training:
            trainer = pl.Trainer(
                enable_checkpointing=True,
                logger=False,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                strategy="ddp_find_unused_parameters_False" if isinstance(self.config.devices, int) and self.config.devices > 1 else "auto",
                devices=find_usable_cuda_devices(self.config.devices) if isinstance(self.config.devices, int) else self.config.devices,
                max_epochs=self.config.max_epochs,
                precision=self.config.precision,
                num_sanity_val_steps=0,
                callbacks=self.config.get_train_callbacks(),
                val_check_interval=self.config.val_check_interval,
                # limit_train_batches=1,
                limit_val_batches=self.config.limit_val_batches,
                # check_val_every_n_epoch=self.config.max_epochs,
            )
        else:
            trainer = pl.Trainer(
                enable_checkpointing=False,
                logger=False,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=find_usable_cuda_devices(1) if isinstance(self.config.devices, int) else self.config.devices,
                precision=self.config.precision,
                num_sanity_val_steps=0,
                callbacks=self.config.get_val_callbacks(),
                # limit_val_batches=100,
            )
        return trainer


class Solution(IModelTrainer):
    def __init__(
            self,
            config,
            data_pipeline: IDataPreparation,
            model_pipeline: IModelTrainer,
    ):
        super().__init__(config)
        self.data_pipeline = data_pipeline
        self.model_pipeline = model_pipeline
        self.current_fold_id = 0

    def train(self, model=None, datamodule=None, *args, **kwargs):
        if datamodule is None:
            datamodule = self.data_pipeline.configure_fold_datamodule(self.current_fold_id)
        if model is None:
            model = self.model_pipeline.configure_model(self.config.model_checkpoint, fold_id=self.current_fold_id)
        model = self.model_pipeline.fit(model, datamodule)
        return defaultdict(lambda: None, model=model, datamodule=datamodule)

    def validate(self, model=None, datamodule=None, *args, **kwargs):
        if datamodule is None:
            datamodule = self.data_pipeline.configure_fold_datamodule(self.current_fold_id)
        if model is None:
            model = self.model_pipeline.configure_model(self.config.model_checkpoint, fold_id=self.current_fold_id)
        val_pred = self.model_pipeline.predict(model, datamodule)
        return defaultdict(lambda: None, model=model, datamodule=datamodule)

    def fit(self, reset_model_checkpoint=True, *args, **kwargs):
        if self.config.training:
            outputs = self.train()
            if reset_model_checkpoint:
                self.config.model_checkpoint = None
        else:
            outputs = defaultdict(lambda: None)
        if self.config.model_checkpoint is None:
            self.config.model_checkpoint = self.model_pipeline.find_best_checkpoint()
        if self.config.validate:
            if self.config.model_checkpoint is not None:
                outputs.pop("model", None)
            print(f"Validate using the best model: {self.config.model_checkpoint}")
            val_outputs = self.validate(**outputs)
            outputs.update(val_outputs)
        if self.config.save_model:
            self.model_pipeline.save_model(outputs["model"], self.current_fold_id)
        if reset_model_checkpoint:
            self.config.model_checkpoint = None
        return outputs

    def cross_validate(self, *args, **kwargs):
        self.current_fold_id = self.config.start_fold
        parent_log_dir_name = self.config.log_dir_name
        outputs_list = []
        while self.current_fold_id < self.config.k_folds:
            self.config.log_dir_name = os.path.join(parent_log_dir_name, f"fold_{self.current_fold_id}")
            os.makedirs(self.config.log_dir, exist_ok=True)
            print(f"fitting fold {self.current_fold_id}")
            outputs = self.fit()
            outputs_list.append(outputs)
            if self.config.single_fold:
                break
            self.current_fold_id += 1
        self.config.log_dir_name = parent_log_dir_name
        return outputs_list
