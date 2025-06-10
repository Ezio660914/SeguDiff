# -*- coding: utf-8 -*-
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ConcordanceIndexLoss(torch.nn.Module):
    def __init__(self, sigma: float):
        """
        Differentiable version of concordance index loss.

        Args:
            sigma: Smoothing parameter for sigmoid approximation (default: 0.1)
                  Lower values make the approximation closer to the true step function
                  but might make gradients more unstable
        """
        super().__init__()
        self.sigma = sigma

    def _smooth_compare(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Smooth approximation of indicator function using sigmoid
        Returns ~1 when x1 > x2, ~0 when x1 < x2
        """
        return torch.sigmoid((x1 - x2) / self.sigma)

    def forward(self, times: torch.Tensor, scores: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        """
        Compute concordance index loss

        Args:
            times: survival times tensor of shape (N,)
            scores: predicted risk scores tensor of shape (N,)
            events: event indicators tensor of shape (N,), 1 if event occurred, 0 if censored

        Returns:
            Loss value (1 - concordance_index) to minimize
        """
        n = times.size(0)

        # Create all possible pairs using upper triangular indices
        idx_i, idx_j = torch.triu_indices(n, n, offset=1)

        # Get pair values
        times_i = times[idx_i]
        times_j = times[idx_j]
        scores_i = scores[idx_i]
        scores_j = scores[idx_j]

        events_i = events[idx_i]
        events_j = events[idx_j]

        coeff = (times_i > times_j).int()

        # time_diffs = torch.abs(times_i - times_j)
        # max_diff = torch.max(time_diffs)
        # time_diffs = time_diffs / max_diff

        # Determine comparable pairs
        # For each pair (i,j), i should have an event and time_i < time_j
        cond_1 = (events_i == 1) & (events_j == 1) & (times_i != times_j)
        cond_2 = (events_i == 1) & (events_j == 0) & (times_i < times_j)
        cond_3 = (events_i == 0) & (events_j == 1) & (times_i > times_j)

        comparable = (cond_1 | cond_2 | cond_3).float()
        # comparable *= time_diffs

        # Compute concordance for each comparable pair
        # Higher risk score should correspond to lower survival time
        concordant = coeff * self._smooth_compare(scores_i, scores_j) + (1 - coeff) * self._smooth_compare(scores_j, scores_i)

        # Compute final concordance index
        numerator = (comparable * concordant).sum()
        denominator = comparable.sum()

        # Add small epsilon to prevent division by zero
        c_index = numerator / (denominator + 1)

        # Return loss (1 - c_index) since we want to maximize c_index
        return c_index
