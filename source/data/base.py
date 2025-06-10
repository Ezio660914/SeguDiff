# -*- coding: utf-8 -*-
import copy
import os
import sys
from enum import auto
import lightning.pytorch as pl
import numpy as np
import torch.utils.data as td
from scipy.signal import lfilter, butter, iirnotch
from scipy.ndimage import median_filter, convolve1d
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

import yaml
import pandas as pd

from ml_utils.utils.array_tools import get_windowed_data, make_folds
from ml_utils.utils.misc import AutoStrEnum
from ml_utils.utils.path_tools import get_stem

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DatasetName(AutoStrEnum):
    mit_bih = auto()
    ludb = auto()


class AbstractDatabase:
    def __init__(self, file_list, window_size, window_sep):
        self.file_list = file_list
        self.window_size = window_size
        self.window_sep = window_sep if window_sep is not None else self.window_size
        self.folds = None
        self.data_df_dict = None
        self.info = None

    def get_base_info(self, file_name, data):
        row_index = np.arange(data.shape[0]).reshape(-1, 1)
        data_with_index = np.concatenate((row_index, data), axis=1)
        windowed_data = get_windowed_data(data_with_index, self.window_size, self.window_sep)
        info = pd.DataFrame({
            'index': windowed_data[:, 0, 0].astype(int),
            'segment_index': np.arange(windowed_data.shape[0]),
        })
        info['window_size'] = self.window_size
        info['window_sep'] = self.window_sep
        info['file_name'] = get_stem(file_name)
        return windowed_data, info

    def make_folds(self, k_folds, *args, **kwargs):
        file_name_dict = {get_stem(f): f for f in self.file_list}
        self.folds = make_folds(list(file_name_dict.keys()), k_folds, True)
        return self.folds

    def save_folds(self, file_path):
        if self.folds is not None:
            with open(file_path, 'w') as f:
                yaml.safe_dump(self.folds, f)

    def load_folds(self, file_path):
        with open(file_path) as f:
            self.folds = yaml.safe_load(f)
        return self.folds

    def preprocess_all(self, *args, **kwargs):
        pass

    def get_dataset(self, *args, **kwargs):
        pass


class DataModuleBase(pl.LightningDataModule):
    def __init__(self, train_dataset=None, val_dataset=None, train_batch_size=32, val_batch_size=None):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else train_batch_size

    def train_dataloader(self):
        if self.train_dataset is not None:
            return td.DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=False)

    def val_dataloader(self):
        if self.val_dataset is not None:
            return td.DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=min(os.cpu_count(), 32), pin_memory=True, persistent_workers=False)


class SegmentIterator(Dataset):
    def __init__(self, data_df_dict, info, dataset_name, used_channels, used_segment_channels=None, ecg_segment_dict=None, ecg_noisy_segment_dict=None, loop=False):
        self.data_df_dict = data_df_dict
        self.info = info
        self.dataset_name = dataset_name
        self.used_channels = used_channels
        self.used_segment_channels = used_segment_channels
        self.ecg_segment_dict = ecg_segment_dict
        self.ecg_noisy_segment_dict = ecg_noisy_segment_dict
        self.loop = loop

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        if self.loop:
            index = index % len(self.info)
        info = self.info.iloc[index]
        file_name = info['file_name']
        df = self.data_df_dict[file_name]
        start = info['index']
        end = start + info['window_size']
        data_dict = {}
        if self.ecg_segment_dict is None:
            ecg = df.loc[start:end - 1, self.used_channels].values.astype(np.float32)
        else:
            segment_index = info['segment_index']
            ecg = self.ecg_segment_dict[file_name][segment_index]
            ecg = ecg[:, self.used_segment_channels].astype(np.float32)
        data_dict['ecg'] = ecg
        if self.ecg_noisy_segment_dict is not None:
            segment_index = info['segment_index']
            ecg_noisy = self.ecg_noisy_segment_dict[file_name][segment_index]
            ecg_noisy = ecg_noisy[:, self.used_segment_channels].astype(np.float32)
            data_dict['ecg_noisy'] = ecg_noisy
        if self.dataset_name == DatasetName.mit_bih:
            data_dict['labels'] = df.loc[start:end - 1, ['label']].values.astype(np.float32)
            if 'r_peak' in df.columns:
                data_dict['r_peak'] = df.loc[start:end - 1, ['r_peak']].values.astype(np.float32)
        elif self.dataset_name == DatasetName.ludb:
            data_dict['pqrst'] = df.loc[start:end - 1, [x + "_label" for x in self.used_channels]].values.astype(np.float32)
        # else:
        #     raise ValueError(f"Dataset {self.dataset_name} not supported")
        return data_dict, info.to_dict()

    @property
    def window_size(self):
        return self.info.iloc[0]['window_size']

    @property
    def window_sep(self):
        return self.info.iloc[0]['window_sep']


def add_noise(ecg, noise, snr, axis):
    k = np.sqrt(np.sum(ecg ** 2, axis, keepdims=True) / (np.sum(noise ** 2, axis, keepdims=True) * np.power(10, snr / 10)))
    ecg_noisy = ecg + k * noise
    return ecg_noisy


class SampleWisePreprocessDataset(Dataset):
    def __init__(self, ecg_dataset=None, noise_dataset=None, snr=0, preprocess=False, repeat_ecg=0):
        """
        Args:
            ecg_dataset:
            noise_dataset:
            snr:
            preprocess:
            repeat_ecg: ecg重复次数，适用于ecg短于噪声数据集的情况
        """
        self.ecg_dataset = ecg_dataset
        self.noise_dataset = noise_dataset
        self.snr = snr
        self.repeat_ecg = repeat_ecg
        self.preprocessors = [
            Normalize(),
        ] if preprocess else []

    @staticmethod
    def prefilter_ecg(ecg_dataset):
        # 后续会修改data_df_dict，需先浅拷贝，防止外部的ecg_dataset也被修改
        ecg_dataset = copy.copy(ecg_dataset)
        if ecg_dataset.dataset_name == DatasetName.mit_bih:
            cols = ['data_1', 'data_2']
            fs = 360
            power_line_freq = 60
            low_pass_filter = 100
        elif ecg_dataset.dataset_name == DatasetName.ludb:
            cols = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
            fs = 360
            power_line_freq = 60
            low_pass_filter = 100
        else:
            raise ValueError(f"Dataset {ecg_dataset.dataset_name} not supported")

        def _prefilter(df):
            ecg = df.loc[:, cols].values
            # 对原始信号进行初步滤波
            # 低通滤波
            b, a = butter(9, [low_pass_filter], "lowpass", fs=fs)
            ecg = lfilter(b, a, ecg, 0)
            # 去除50 or 60Hz工频
            b, a = iirnotch(power_line_freq, power_line_freq, fs=fs)
            ecg = lfilter(b, a, ecg, 0)
            # 去除基线漂移
            ecg = ecg_blw_removal_meanfilt(ecg, fs, 0)
            return ecg

        prefiltered_data_df_dict = {}
        for file_name, df in tqdm(ecg_dataset.data_df_dict.items(), desc="prefilter ecg"):
            ecg = _prefilter(df)
            df = df.copy()
            df.loc[:, cols] = ecg
            prefiltered_data_df_dict[file_name] = df
        ecg_dataset.data_df_dict = prefiltered_data_df_dict
        return ecg_dataset

    def __len__(self):
        return len(self.ecg_dataset) * (self.repeat_ecg + 1)

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index
        if index >= len(self) or index < 0:
            raise IndexError("value of index should not exceed dataset length")
        n, i = divmod(index, len(self.ecg_dataset))
        data_dict, info = self.ecg_dataset[i]
        ecg = data_dict['ecg']
        for p in self.preprocessors:
            p.fit(ecg, 0)
            ecg = p.transform(ecg)
        data_dict['ecg'] = ecg
        if self.noise_dataset is not None:
            # 当ecg数据集repeat时，噪声数据的index要变
            noise_index = info['index']
            noise_index += n * len(self.ecg_dataset.data_df_dict[info['file_name']])
            data_dict['noise_index'] = noise_index
            noise = self.noise_dataset[noise_index]
            ecg_noisy = add_noise(ecg, noise, snr=self.snr, axis=0)
            data_dict['ecg_noisy'] = ecg_noisy.astype(np.float32)
        return data_dict, info


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


def ecg_blw_removal_medfilt(ecg, fs, axes=None):
    window1 = 0.2
    window2 = 0.6
    # remove P & QRS
    bl = median_filter(ecg, size=int(fs * window1 // 2 * 2 + 1), axes=axes)
    # remove T
    bl = median_filter(bl, size=int(fs * window2 // 2 * 2 + 1), axes=axes)
    return ecg - bl


def mean_filter_1d(signal, kernel_size, axis=0):
    """
    一维均值滤波实现
    :param signal: 输入的一维信号
    :param kernel_size: 滤波窗口大小
    :return: 滤波后的信号
    """
    kernel = np.ones(kernel_size) / kernel_size
    return convolve1d(signal, kernel, axis)


def ecg_blw_removal_meanfilt(ecg, fs, axis=0):
    """
    使用均值滤波移除心电信号的基线漂移
    :param ecg: 输入的心电信号
    :param fs: 采样频率
    :return: 去除基线漂移后的心电信号
    """
    window1 = 0.2  # 窗口1的时间长度
    window2 = 0.6  # 窗口2的时间长度
    # 使用均值滤波去除 P & QRS
    bl = mean_filter_1d(ecg, int(fs * window1 // 2 * 2 + 1), axis)
    # 使用均值滤波去除 T
    bl = mean_filter_1d(bl, int(fs * window2 // 2 * 2 + 1), axis)
    return ecg - bl
