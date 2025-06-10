# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class IModelTrainer:
    def __init__(self, config, **hyperparams):
        self.config = config
        self.hyperparams = hyperparams

    def configure_model(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def validate(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def cross_validate(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def save_model(self, *args, **kwargs):
        raise NotImplementedError

    def find_best_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def find_last_checkpoint(self, *args, **kwargs):
        raise NotImplementedError

    def get_model_checkpoints(self):
        raise NotImplementedError


class IModelInference:
    def __init__(self, config, **hyperparams):
        self.config = config
        self.hyperparams = hyperparams

    def prepare_data(self, *args, **kwargs):
        pass

    def validate(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError
