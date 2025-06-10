# -*- coding: utf-8 -*-
import glob
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchmetrics.functional as tmf
import tqdm

from ml_utils.metrics.snr import SignalNoiseRatio
from ml_utils.utils.mp import async_run
from ml_utils.utils.path_tools import get_stem

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

metrics_func = {
    'rmse': partial(tmf.mean_squared_error, squared=False),
    'snr': SignalNoiseRatio(),
}


def file_metrics_denoise(result_files, save_dir, pred_col_names, target_col_names, use_async_run=True):
    def _process(file):
        df = pd.read_csv(file)
        df = df.dropna()

        preds = torch.as_tensor(df[pred_col_names].values, dtype=torch.float32).ravel()
        targets = torch.as_tensor(df[target_col_names].values, dtype=torch.float32).ravel()
        metrics_dict = {'file_name': get_stem(file)}
        for name, metrics in metrics_func.items():
            metrics_dict[name] = metrics(preds, targets).item()
        return metrics_dict

    if use_async_run:
        result_list = async_run(_process, result_files, lambda x: (x,), desc="Process file metrics")
    else:
        result_list = [_process(file) for file in tqdm.tqdm(result_files, desc="Process file metrics")]

    result_df = pd.DataFrame.from_records(result_list)
    print(result_df)
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "file_metrics.csv"), index=False)


def overall_metrics_denoise(result_files, save_dir, pred_col_names, target_col_names, use_async_run=True):
    if use_async_run:
        df = async_run(pd.read_csv, result_files, lambda x: (x,), "loading csv")
    else:
        df = [pd.read_csv(file) for file in tqdm.tqdm(result_files, "loading csv")]
    df = pd.concat(df, ignore_index=True)
    df = df.dropna()
    preds = torch.as_tensor(df[pred_col_names].values, dtype=torch.float32).ravel()
    targets = torch.as_tensor(df[target_col_names].values, dtype=torch.float32).ravel()
    metrics_df = {}
    for name, metrics in metrics_func.items():
        metrics_df[name] = metrics(preds, targets).item()
    metrics_df = pd.Series(metrics_df, name="metrics")
    print(metrics_df)
    os.makedirs(save_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(save_dir, "overall_metrics.csv"))


def file_segment_metrics_denoise(result_files, save_dir, pred_col_names, target_col_names, window_size=1024, use_async_run=True):
    def get_file_segment_metrics(file):
        df = pd.read_csv(file)
        # 训练时用的无重叠窗口，可直接reshape
        preds = np.asarray(df[pred_col_names].values, dtype=np.float32)[:len(df) // window_size * window_size].reshape(-1, window_size)
        targets = np.asarray(df[target_col_names].values, dtype=np.float32)[:len(df) // window_size * window_size].reshape(-1, window_size)
        mask = ~np.any(np.isnan(preds), axis=1)
        preds, targets = preds[mask], targets[mask]
        # print(preds.shape, targets.shape)
        metrics_df = {}
        metrics_df['rmse'] = np.sqrt(np.mean((preds - targets) ** 2, 1))
        metrics_df['snr'] = 10 * np.log10(np.sum(targets ** 2, 1) / np.sum((preds - targets) ** 2, 1))
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.insert(0, "index", np.arange(len(mask))[mask, None])
        metrics_df.insert(0, "file_name", get_stem(file))
        return metrics_df

    if use_async_run:
        df_list = async_run(get_file_segment_metrics, result_files, lambda x: (x,), "get file segment metrics")
    else:
        df_list = [get_file_segment_metrics(file) for file in tqdm.tqdm(result_files, "get file segment metrics")]
    df_list = pd.concat(df_list, ignore_index=True)
    print(df_list)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        df_list.to_csv(os.path.join(save_dir, "file_segment_metrics.csv"))
    plt.hist(df_list["snr"], bins=50)
    plt.show()
    return df_list
