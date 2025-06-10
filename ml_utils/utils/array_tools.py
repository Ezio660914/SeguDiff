# -*- coding: utf-8 -*-
import os
import random
import sys

import numpy as np
import torch
from torch.utils import data as td

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def split_array(array, ratio=0.8, axis=0):
    """split array according to ratio"""
    return np.split(array, [int(ratio * array.shape[axis])], axis)


def create_tensor_dataset(*arrays):
    tensors = []
    for a in arrays:
        tensors.append(torch.from_numpy(a).float())
    return td.TensorDataset(*tensors)


def chunks(x: list, k, shuffle=True):
    """
    split list by k

    :param x:
    :param k:
    :return:
    """
    if shuffle:
        x = x.copy()
        random.shuffle(x)
    n, m = divmod(len(x), k)
    return [x[i * n + min(i, m):(i + 1) * n + min(i + 1, m)] for i in range(k)]


def make_folds(x: list, k=5, shuffle=True):
    ck = chunks(x, k, shuffle)
    folds = []
    for i in range(k):
        train_fold = []
        for j, c in enumerate(ck):
            if not j == i:
                train_fold += c
        folds.append((train_fold, ck[i]))
    return folds


def split_list(x, ratio=0.8):
    i = int(len(x) * ratio)
    return x[:i], x[i:]


def tensor_to_array(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_scalar(t):
    if isinstance(t, np.ndarray):
        assert t.ndim == 0
        return t.item()
    elif isinstance(t, torch.Tensor):
        assert t.ndim == 0
        return t.detach().cpu().item()
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def get_windowed_data(time_series, window_size=30, window_sep=1):
    """
    划分时间序列窗口

    :param time_series:
    :param window_size: 窗口大小
    :param window_sep: 窗口间距
    :return:
    """
    row = np.expand_dims(np.arange(window_size), 0)
    column = np.expand_dims(np.arange(time_series.shape[0] - window_size + 1, step=window_sep), 1)
    index = row + column
    return time_series[index]
