# -*- coding: utf-8 -*-
import os
import sys

import lightning.pytorch as pl
import torch.utils.data as td
import torch
import numpy as np
from scipy import signal

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, predict_dataset=None, train_batch_size=32, val_batch_size=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size

    def train_dataloader(self):
        if self.train_dataset is not None:
            return td.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=False)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return td.DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=False)

    def predict_dataloader(self):
        if self.predict_dataset is not None:
            return td.DataLoader(self.predict_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=False)


class Preprocessor:
    def fit(self, *args, **kwargs):
        pass

    def transform(self, *args, **kwargs):
        pass

    def reverse(self, *args, **kwargs):
        pass


class ZeroMean(Preprocessor):
    def __init__(self):
        self.bias = None

    def fit(self, data, dim=1):
        """
        data: [B,L,C]
        """
        if isinstance(data, torch.Tensor):
            self.bias = torch.mean(data, dim=dim, keepdim=True)
        else:
            self.bias = np.mean(data, axis=dim, keepdims=True)

    def transform(self, data):
        return data - self.bias

    def reverse(self, data):
        return data + self.bias


class Normalize(Preprocessor):
    def __init__(self):
        self.data_min = None
        self.data_max = None
        self.div = None

    def fit(self, data, dim=1):
        """
        data: [B,L,C]
        """
        if isinstance(data, torch.Tensor):
            self.data_min = torch.min(data, dim=dim, keepdim=True).values
            self.data_max = torch.max(data, dim=dim, keepdim=True).values
        else:
            self.data_min = np.min(data, dim, keepdims=True)
            self.data_max = np.max(data, dim, keepdims=True)
        self.div = self.data_max - self.data_min
        self.div[self.div == 0] = 1e-12

    def transform(self, data):
        return (data - self.data_min) / self.div

    def reverse(self, data):
        return data * (self.data_max - self.data_min) + self.data_min


class Scale(Preprocessor):
    def __init__(self, multiplier, bias):
        self.multiplier = multiplier
        self.bias = bias

    def transform(self, data):
        return (data * self.multiplier) + self.bias

    def reverse(self, data):
        return (data - self.bias) / self.multiplier


class Resample(Preprocessor):
    def __init__(self, fs, fs_target):
        self.fs = fs
        self.fs_target = fs_target
        self.new_length = None
        self.dim = None

    def fit(self, data, dim=0):
        self.new_length = int(data.shape[dim] * self.fs_target / self.fs)
        self.dim = dim

    def transform(self, data):
        data = signal.resample(data, self.new_length, axis=self.dim)
        return data
