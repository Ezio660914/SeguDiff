# -*- coding: utf-8 -*-
import os
import sys

import torch
import yaml
from lightning import pytorch as pl
from lightning_fabric.accelerators import find_usable_cuda_devices
from torch.distributed import is_initialized, get_rank

from ml_utils.experiments.base.data_pipeline import IDataPreparation, IResultSubmission
from ml_utils.experiments.base.model_pipeline import IModelTrainer
from ml_utils.utils.register import Register
from source.trainer.base import AbstractTrainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ExperimentPipeline(IModelTrainer):
    def __init__(self, config, data_pipeline: IDataPreparation, model_pipeline: IModelTrainer, result_pipeline: IResultSubmission = None, **hyperparams):
        super().__init__(config, **hyperparams)
        self.data_pipeline = data_pipeline
        self.model_pipeline = model_pipeline
        self.result_pipeline = result_pipeline
        self.current_fold_id = 0

    def configure_hyperparams(self, args=None):
        self.config.parse_args(args)
        os.makedirs(self.config.log_dir, exist_ok=True)
        print(vars(self.config))

    def train(self, model=None, datamodule=None):
        if datamodule is None:
            datamodule = self.data_pipeline.configure_fold_datamodule(fold_id=self.current_fold_id)
        if model is None:
            model = self.model_pipeline.configure_model(model_checkpoint=self.config.model_checkpoint, fold_id=self.current_fold_id)
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
        if self.config.model_checkpoint is not None and os.path.exists(self.config.model_checkpoint) and self.config.resume_from_checkpoint:
            ckpt_path = self.config.model_checkpoint
        else:
            ckpt_path = None
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        return model, datamodule

    def validate(self, model=None, datamodule=None):
        if datamodule is None:
            datamodule = self.data_pipeline.configure_fold_datamodule(fold_id=self.current_fold_id)
        if model is None:
            model = self.model_pipeline.configure_model(model_checkpoint=self.config.model_checkpoint, fold_id=self.current_fold_id)
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
        trainer.validate(model, datamodule=datamodule)
        return model, datamodule

    def fit(self, reset_model_checkpoint=True):
        model = None
        datamodule = None
        if self.config.training:
            model, datamodule = self.train()
            if reset_model_checkpoint:
                self.config.model_checkpoint = None
        if not is_initialized() or (is_initialized() and get_rank() == 0):
            if self.config.model_checkpoint is None:
                self.config.model_checkpoint = self.config.find_best_checkpoint()
            if self.config.validate:
                print(f"Validate using the best model: {self.config.model_checkpoint}")
                model, datamodule = self.validate(datamodule=datamodule)
            if model is not None and self.config.save_torch_model:
                self.save_model(model)
            if reset_model_checkpoint:
                self.config.model_checkpoint = None

    def save_model(self, model: AbstractTrainer, *args, **kwargs):
        os.makedirs(self.config.torch_model_save_dir, exist_ok=True)
        torch_model_file = os.path.join(self.config.torch_model_save_dir, f"torch_model_fold_{self.current_fold_id}.pt")
        model.save_torch_model(torch_model_file)

    def cross_validate(self):
        self.current_fold_id = self.config.start_fold
        parent_log_dir_name = self.config.log_dir_name
        while self.current_fold_id < self.config.k_folds:
            self.config.log_dir_name = os.path.join(parent_log_dir_name, f"fold_{self.current_fold_id}")
            os.makedirs(self.config.log_dir, exist_ok=True)
            self.fit()
            if self.result_pipeline is not None and os.path.exists(self.config.log_file):
                self.result_pipeline.merge_results()
            if self.config.single_fold:
                break
            self.current_fold_id += 1
        self.config.log_dir_name = parent_log_dir_name


class ModelPipeline(IModelTrainer):
    def __init__(self, config, **hyperparams):
        super().__init__(config, **hyperparams)
        with open(self.config.yaml_config) as f:
            self.yaml_config = yaml.safe_load(f)

    def configure_model(self, model_checkpoint, fold_id, *args, **kwargs):
        trainer_class = Register['trainer', self.yaml_config['trainer']['class']]
        trainer_args = self.yaml_config['trainer']['args']
        models = {}
        for name, cfg in self.yaml_config['models'].items():
            model_class = Register['model', cfg['class']]
            model_args = cfg['args']
            model = model_class(**model_args)
            if 'checkpoint' in cfg.keys() and cfg['checkpoint'] is not None:
                model.load_state_dict(torch.load(cfg['checkpoint'], map_location='cpu'))
            models[name] = model
        if model_checkpoint is not None:
            print(f"loading from {model_checkpoint}")
            model = trainer_class.load_from_checkpoint(
                checkpoint_path=model_checkpoint,
                map_location='cpu',
                **models,
                **trainer_args
            )
        else:
            model = trainer_class(**models, **trainer_args)
        self.print_summary(model)
        return model

    def print_summary(self, *args, **kwargs):
        pass
