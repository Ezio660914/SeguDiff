# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import sys

import numpy as np
import torch
from tqdm import trange

from ml_utils.utils.array_tools import to_scalar
from ml_utils.utils.mp import async_run

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _get_result(base_metric, n, size, y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        index_arr = torch.randint(n, size=size)
    else:
        index_arr = np.random.randint(n, size=size)
    label_sample = y_true[index_arr]
    pred_sample = y_pred[index_arr]
    if isinstance(y_true, torch.Tensor):
        result = base_metric(pred_sample, label_sample)  # torchmetrics
    else:
        result = base_metric(label_sample, pred_sample)  # sklearn
    return result


def confidence_interval_bootstrap(y_pred, y_true, base_metric, num_bootstraps=2000, confidence_level=0.95, use_for_loop=True, processes=min(mp.cpu_count(), 32)):
    """
    calculate confidence interval by bootstrap method
    :param y_pred:
    :param y_true:
    :param num_bootstraps: sample times B>=1000 usually
    :param confidence_level: confidence level 0.95 usually
    :param base_metric: metric function
    :return: bootstrap confidence interval lower & upper bound
    """
    n = len(y_true)
    backbone = torch if isinstance(y_true, torch.Tensor) else np
    if use_for_loop:
        if processes is None:
            sample_result = []
            for _ in trange(num_bootstraps, desc='bootstraps'):
                sample_result.append(_get_result(base_metric, n, [n], y_pred, y_true))
        else:
            sample_result = async_run(_get_result, range(num_bootstraps), lambda x: (base_metric, n, [n], y_pred, y_true), desc='bootstraps', processes=processes)
        sample_result = backbone.stack(sample_result)
    else:
        sample_result = _get_result(base_metric, n, [num_bootstraps, n], y_pred, y_true)

    a = 1 - confidence_level
    k1 = int(num_bootstraps * a / 2)
    k2 = int(num_bootstraps * (1 - a / 2))

    sample_result_sorted = sorted(sample_result)
    lower = to_scalar(sample_result_sorted[k1])
    upper = to_scalar(sample_result_sorted[k2])
    print(lower, upper)
    return lower, upper
