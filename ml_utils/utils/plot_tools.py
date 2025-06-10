# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, rgb_to_hsv, to_rgba_array, hsv_to_rgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def cal_subplot_layout(num_plots, base_size=5):
    """
    Calculate the best number of rows and columns of the graph from the total number of subplots, and figure size
    """
    row = np.sqrt(num_plots).astype(int)
    col = np.ceil(num_plots / row).astype(int)
    figsize = np.asarray(base_size) * np.asarray([col, row])
    return row, col, figsize


def create_paired_colors(base_cm, n_color, n_pair, min_hue_step=0., min_saturation_step=0.1, min_value_step=0.1):
    hue_step = min(min_hue_step, 1 / n_pair)
    saturation_step = min(min_saturation_step, 1 / n_pair)
    value_step = min(min_value_step, 1 / n_pair)
    color = base_cm(np.arange(base_cm.N))
    cm = LinearSegmentedColormap.from_list("color", color, n_color)
    # shape of [N, 4]
    color = cm(np.arange(n_color))
    # [N,3]
    color_hsv = rgb_to_hsv(color[:, :3])
    color_list = []
    for i in range(n_pair):
        modifier = np.array([i * hue_step, i * saturation_step, i * value_step]).reshape(1, 3)
        new_color_hsv = np.clip(color_hsv + modifier, 0, 1)
        color_list.append(new_color_hsv)
    color_hsv = np.concatenate(color_list, axis=0)
    color = to_rgba_array(hsv_to_rgb(color_hsv), 1)
    return color
