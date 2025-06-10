# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def triplesurv_ranking_loss(risks, durations, events, sigma=1, rou=1, clip=False, use_activation=True, eps=1e-8):
    """

    Args:
        risks: [batch, 1]
        durations: [batch, 1]
        events: [batch, 1], 0=censored, 1=event
        sigma: float scalar
        rou: float scalar for duration
    Returns:

    """
    delta_risks = risks - risks.T
    delta_durations = durations.T - durations
    loss = rou * delta_durations - delta_risks
    # 是否在risk的差值大于duration的差值时，认为二者距离已经足够，并忽略loss
    if clip:
        loss = torch.relu(loss)
    if use_activation:
        loss = torch.exp(sigma * loss)
    mask = events * torch.relu(torch.sign(delta_durations))
    loss = torch.sum(mask * loss) / (torch.sum(mask) + eps)
    return loss
