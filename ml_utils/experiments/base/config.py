# -*- coding: utf-8 -*-
import argparse
import glob
import os
import platform
import sys

import torch

from ml_utils.utils.path_tools import get_project_dir

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if platform.system() == "Linux":
    pass
    # matplotlib.use('agg')


class Config:
    def __init__(self, **hyperparams):
        # basic path
        self.project_dir = get_project_dir()
        self.data_dir = os.path.join(self.project_dir, "data")
        self.precision = 32
        self.num_workers = min(os.cpu_count(), 32)
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hyperparams = hyperparams

    def parse_args(self, args=None):
        ap = argparse.ArgumentParser()
        # ap.add_argument()
        ap.parse_args(args, namespace=self)

    @property
    def log_dir(self):
        return ""

    @property
    def yaml_dir(self):
        return os.path.join(self.project_dir, "yaml")

    @property
    def model_save_dir(self):
        return self.log_dir

    def get_train_callbacks(self):
        return []

    def get_val_callbacks(self):
        return []

    def find_best_checkpoint(self, file_name="model*.pt"):
        """
        在log dir中查找性能最佳的模型检查点

        Returns:

        """
        model_files = glob.glob(os.path.join(self.log_dir, file_name))
        if len(model_files) == 0:
            return None
        else:
            return max(model_files, key=os.path.getctime)

    def find_last_checkpoint(self, file_name="*.pt"):
        """
        在log dir中查找最近一次保存的模型
        """
        model_files = glob.glob(os.path.join(self.log_dir, file_name))
        if len(model_files) == 0:
            return None
        else:
            return max(model_files, key=os.path.getctime)

    def init(self, args=None):
        self.parse_args(args)
        os.makedirs(self.log_dir, exist_ok=True)
        print(vars(self))
