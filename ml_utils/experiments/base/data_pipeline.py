# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class IDataPreparation:
    def __init__(self, config, **hyperparams):
        self.config = config
        self.hyperparams = hyperparams
        self.database = None

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    def configure_fold_datamodule(self, *args, **kwargs):
        raise NotImplementedError

    def configure_test_datamodule(self, *args, **kwargs):
        raise NotImplementedError


class IResultSubmission:
    def __init__(self, config):
        self.config = config

    def get_results(self, *args, **kwargs):
        raise NotImplementedError

    def merge_results(self, *args, **kwargs):
        raise NotImplementedError

    def prepare_for_submission(self):
        raise NotImplementedError
