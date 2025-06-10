# -*- coding: utf-8 -*-
import os
import sys
from scipy.ndimage import median_filter, convolve1d
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


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
