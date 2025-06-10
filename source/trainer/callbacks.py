# -*- coding: utf-8 -*-
import os
import sys
from typing import Any

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ml_utils.utils.plot_tools import create_paired_colors

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class VisualizeSegmentation(pl.Callback):
    def __init__(self, data_keys=None, mask_keys=None, label_names=None, background_index=(0,), n_samples=1, n_batches=1, log_dir=None):
        self.data_keys = data_keys
        self.mask_keys = mask_keys
        self.label_names = label_names
        self.background_index = background_index
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        # cm = ListedColormap(["#403990", "#80A6E2", "#FBDD85", "#F46F43", "#CF3D3E"])
        cm = plt.get_cmap("Set3")
        self.color = create_paired_colors(
            cm, len(label_names), len(mask_keys),
            0.02, 0.3, 0.1
        ).reshape(len(mask_keys), len(label_names), 4)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.n_batches is not None and batch_idx >= self.n_batches:
            return
        data_dict, info = batch
        batch_data_0 = next(iter(data_dict.values()))
        batch_size, seg_len, n_channels = batch_data_0.shape
        index = np.random.permutation(batch_size)[:self.n_samples if self.n_samples >= 0 else batch_size]
        if self.data_keys is not None and self.mask_keys is not None:
            for n, idx in enumerate(index):
                plt.figure(figsize=(10, 5 * n_channels))
                for i in range(n_channels):
                    plt.subplot(n_channels, 1, i + 1)
                    plt.title("Channel {}".format(i + 1))
                    for name in self.data_keys:
                        if name in outputs.keys():
                            plt.plot(outputs[name][idx, :, i], label=name)
                    for j, name in enumerate(self.mask_keys):
                        if name in outputs.keys():
                            self._plot_masks(outputs[name][idx], name, self.color[j])
                handles, labels = plt.gca().get_legend_handles_labels()
                unique_index = np.unique(labels, return_index=True)[1]
                handles, labels = list(zip(*[(handles[ui], labels[ui]) for ui in unique_index]))
                plt.legend(handles, labels, frameon=False, loc=(1.01, 0.5))
                plt.tight_layout()
                if self.log_dir is not None:
                    plt.savefig(os.path.join(self.log_dir, f"epoch_{trainer.current_epoch + 1}_step_{trainer.global_step}_batch_{batch_idx + 1}_example_{n + 1}_channel.png"))
                plt.show()

    def _plot_masks(self, mask, name, color):
        # mask: shape of [segment length, num classes]
        if mask.shape[1] > 1:
            label = mask.argmax(1)
        else:
            label = mask.squeeze(1).astype('int')
        split = np.concatenate([[0], np.where(np.diff(label) != 0)[0] + 1, [len(label)]])
        for start, end in zip(split[:-1], split[1:]):
            l = label[start:end][0]
            if l not in self.background_index:
                edgecolor = color[l].copy()
                facecolor = color[l].copy()
                # alpha
                edgecolor[3] = 1
                facecolor[3] = 0.2
                plt.axvspan(start, end, edgecolor=edgecolor, facecolor=facecolor, linestyle='--', linewidth=2, label=' '.join([name, self.label_names[l]]))
