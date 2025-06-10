# -*- coding: utf-8 -*-
import argparse
import os
import sys
import platform
from pathlib import Path
import glob
import matplotlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if platform.system() == "Linux":
    pass
    # matplotlib.use('agg')


def get_project_dir(project_root="SeguDiff"):
    file_path = Path(os.path.abspath(__file__))
    while file_path.name != project_root:
        file_path = file_path.parent
        if file_path == file_path.parent:
            break
    return file_path


class Config:
    def __init__(self):
        # basic path
        self.project_dir = get_project_dir()
        self.data_dir = os.path.join(self.project_dir, "data")

        self.precision = 32
        self.num_workers = min(os.cpu_count(), 32)

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
    def torch_model_save_dir(self):
        return self.log_dir

    def get_train_callbacks(self):
        return []

    def get_val_callbacks(self):
        return []

    def find_best_checkpoint(self):
        """
        在log dir中查找性能最佳的模型检查点

        Returns:

        """
        model_files = glob.glob(os.path.join(self.log_dir, "model*.ckpt"))
        if len(model_files) == 0:
            return None
        else:
            return max(model_files, key=os.path.getctime)

    def find_last_checkpoint(self):
        """
        在log dir中查找最近一次保存的模型
        """
        model_files = glob.glob(os.path.join(self.log_dir, "*.ckpt"))
        if len(model_files) == 0:
            return None
        else:
            return max(model_files, key=os.path.getctime)
