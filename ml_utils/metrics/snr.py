# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SignalNoiseRatio(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        snr = 10 * torch.log10(torch.sum(target ** 2) / torch.sum((preds - target) ** 2))
        return snr
