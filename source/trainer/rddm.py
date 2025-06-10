# -*- coding: utf-8 -*-
import os
import sys
from functools import partial
from typing import Literal
import math
import torch
import torch.nn.functional as F
from einops import reduce
from tqdm import tqdm
import numpy as np

from source.trainer.base import AbstractTrainer
from ml_utils.utils.register import Register
from ml_utils.utils.array_tools import tensor_to_array, extract

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def gen_coefficients(timesteps, schedule: Literal['increased', 'decreased', 'average', 'normal'] = "increased", sum_scale=1., ratio=1.):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x ** ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y / y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x ** ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y / y_sum
    elif schedule == "average":
        alphas = torch.full([timesteps], 1 / timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3 + mu, 3 + mu, timesteps, dtype=np.float32)
        y = np.e ** (-((x - mu) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * (sigma ** 2))
        y = torch.from_numpy(y)
        alphas = y / y.sum()
    else:
        raise ValueError("Unknown schedule")
    assert (alphas.sum() - 1).abs() < 1e-6

    return alphas * sum_scale


@Register(group_name="trainer", func_name="RDDM")
class RDDM(AbstractTrainer):
    def __init__(
            self,
            torch_model,
            n_diffusion_steps=1000,
            n_sampling_steps=None,
            objective: Literal['pred_res', 'pred_noise', 'pred_res_noise'] = 'pred_res',
            ddim_sampling_eta=0,
            sum_scale=0.01,
            alpha_ratio=1,
            beta_ratio=1,
            channel_last=True,
            cond_drop_prob=0,
            cond_scale=1,
            rescaled_phi=0,
            clip_x_start=(-5, 5),
            alpha_schedule: Literal['increased', 'decreased', 'average', 'normal'] = 'decreased',
            beta_schedule: Literal['increased', 'decreased', 'average', 'normal'] = 'increased',
            min_snr_loss_weight=False,
            min_snr_gamma=700,
            preprocess=False
    ):
        super().__init__(torch_model, preprocess)
        self.n_diffusion_steps = n_diffusion_steps
        self.n_sampling_steps = n_sampling_steps if n_sampling_steps is not None else n_diffusion_steps
        self.objective = objective

        assert self.n_sampling_steps <= n_diffusion_steps
        self.is_ddim_sampling = self.n_sampling_steps < n_diffusion_steps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.sum_scale = sum_scale
        self.channel_last = channel_last
        self.cond_drop_prob = cond_drop_prob
        self.cond_scale = cond_scale
        self.rescaled_phi = rescaled_phi
        self.clip_x_start = clip_x_start

        self.betas2 = gen_coefficients(n_diffusion_steps, beta_schedule, sum_scale, beta_ratio)
        self.betas2_cumsum = self.betas2.cumsum(dim=0).clip(0, 1)
        self.betas2_cumsum_prev = F.pad(self.betas2_cumsum[:-1], (1, 0), value=self.betas2_cumsum[0])

        self.alphas = gen_coefficients(n_diffusion_steps, alpha_schedule, 1, alpha_ratio)
        self.alphas_cumsum = self.alphas.cumsum(dim=0).clip(0, 1)
        self.alphas_cumsum_prev = F.pad(self.alphas_cumsum[:-1], (1, 0), value=self.alphas_cumsum[0])
        self.one_minus_alphas_cumsum = torch.clamp(1 - self.alphas_cumsum, min=1e-8)

        self.betas = torch.sqrt(self.betas2)
        self.betas_cumsum = torch.sqrt(self.betas2_cumsum)

        self.posterior_mean_coef1 = self.betas2_cumsum_prev / self.betas2_cumsum
        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2 = (self.betas2 * self.alphas_cumsum_prev - self.betas2_cumsum_prev * self.alphas) / self.betas2_cumsum
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3 = self.betas2 / self.betas2_cumsum
        self.posterior_mean_coef3[0] = 1

        self.posterior_variance = self.betas2 * self.betas2_cumsum_prev / self.betas2_cumsum
        self.posterior_variance[0] = 0
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        if min_snr_loss_weight:
            snr = self.alphas_cumsum ** 2 / self.betas2_cumsum
            loss_weight = torch.clamp(torch.clamp(snr, max=min_snr_gamma) / snr, min=1e-3)
            loss_weight[0] = 1
        else:
            loss_weight = torch.ones(n_diffusion_steps)
        self.loss_weight = loss_weight

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.betas2 = self.betas2.to(*args, **kwargs)
        self.betas2_cumsum = self.betas2_cumsum.to(*args, **kwargs)
        self.betas2_cumsum_prev = self.betas2_cumsum_prev.to(*args, **kwargs)
        self.alphas = self.alphas.to(*args, **kwargs)
        self.alphas_cumsum = self.alphas_cumsum.to(*args, **kwargs)
        self.alphas_cumsum_prev = self.alphas_cumsum_prev.to(*args, **kwargs)
        self.one_minus_alphas_cumsum = self.one_minus_alphas_cumsum.to(*args, **kwargs)
        self.betas = self.betas.to(*args, **kwargs)
        self.betas_cumsum = self.betas_cumsum.to(*args, **kwargs)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(*args, **kwargs)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(*args, **kwargs)
        self.posterior_mean_coef3 = self.posterior_mean_coef3.to(*args, **kwargs)
        self.posterior_variance = self.posterior_variance.to(*args, **kwargs)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(*args, **kwargs)
        self.loss_weight = self.loss_weight.to(*args, **kwargs)
        return self

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
                (x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_input -
                 extract(self.betas_cumsum, t, x_t.shape) * noise) / extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
                (x_t - x_input - (extract(self.alphas_cumsum, t, x_t.shape) - 1)
                 * pred_res) / extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
                x_t - extract(self.alphas_cumsum, t, x_t.shape) * x_res -
                extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def model_predictions(self, x, t, x_input, cond_scale=6., rescaled_phi=0.7, clip_x_start=(0, 1)):
        model_output = self.forward_with_cond_scale(x, t, cond=x_input, cond_scale=cond_scale, rescaled_phi=rescaled_phi)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            pred_res = x_input - x_start
        elif self.objective == 'pred_res':
            pred_res = model_output
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
        elif self.objective == 'pred_res_noise':
            pred_res, pred_noise = torch.chunk(model_output, 2, 2)
            x_start = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)
        else:
            raise ValueError(f'Unknown objective: {self.objective}')

        if clip_x_start is not None:
            x_start = torch.clamp(x_start, *clip_x_start)

        return pred_res, pred_noise, x_start

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
                extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
                extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_input, cond_scale, rescaled_phi, clip_x_start=(0, 1)):
        pred_res, pred_noise, x_start = self.model_predictions(x, t, x_input, cond_scale, rescaled_phi, clip_x_start)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_res, x_start, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_input, cond_scale=6., rescaled_phi=0.7, clip_x_start=(0, 1)):
        batched_times = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_input, cond_scale, rescaled_phi, clip_x_start)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        x_t_prev = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_t_prev, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_shape=None, clip_x_start=(-1, 1), cond=None, cond_scale=6., rescaled_phi=0.7, return_sequence=False, *args, **kwargs):
        x_t = cond + math.sqrt(self.sum_scale) * torch.randn_like(cond)
        x_list = [x_t] if return_sequence else None
        for t in tqdm(reversed(range(0, self.n_diffusion_steps)), desc='ddpm sampling', total=self.n_diffusion_steps):
            x_t, x_start = self.p_sample(x_t, t, cond, cond_scale, rescaled_phi, clip_x_start)
            if return_sequence:
                x_list.append(x_t)
        return x_list if return_sequence else x_t

    @torch.no_grad()
    def ddim_sample(self, x_shape=None, clip_x_start=(-1, 1), cond=None, cond_scale=6., rescaled_phi=0.7, n_sampling_steps=None, ddim_sampling_eta=None, return_sequence=False, *args, **kwargs):
        n_sampling_steps = self.n_sampling_steps if n_sampling_steps is None else n_sampling_steps
        ddim_sampling_eta = self.ddim_sampling_eta if ddim_sampling_eta is None else ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.n_diffusion_steps - 1,
                               steps=n_sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = cond + math.sqrt(self.sum_scale) * torch.randn_like(cond)
        x_list = [x_t] if return_sequence else None
        for time, time_next in tqdm(time_pairs, desc='ddim sampling'):
            batched_times = torch.full((x_t.shape[0],), time, device=self.device, dtype=torch.long)
            pred_res, pred_noise, x_start = self.model_predictions(x_t, batched_times, cond, cond_scale, rescaled_phi, clip_x_start)

            if time_next < 0:
                x_t = x_start
                continue
            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum - betas2_cumsum_next
            betas_cumsum = self.betas_cumsum[time]
            sigma2 = ddim_sampling_eta * (betas2 * betas2_cumsum_next / betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (betas2_cumsum_next - sigma2).sqrt() / betas_cumsum
            noise = torch.randn_like(x_t)
            x_t = sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum * x_t + \
                  (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start + \
                  (alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                  sigma2.sqrt() * noise
            if return_sequence:
                x_list.append(x_t)
        return x_list if return_sequence else x_t

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(*args, **kwargs)

    def q_sample(self, x_start, x_res, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)

        return (
                x_start + extract(self.alphas_cumsum, t, x_start.shape) * x_res +
                extract(self.betas_cumsum, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, x_input, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_res = x_input - x_start

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise)

        # predict and take gradient step

        model_out = self.forward(x, t, x_input, self.cond_drop_prob)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_res':
            target = x_res
        elif self.objective == 'pred_res_noise':
            target = torch.cat([x_res, noise], 2)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        loss = loss * extract(self.loss_weight, t, loss.shape)
        loss = loss.mean()
        return loss

    def forward(self, x, t, cond, cond_drop_prob=0.5):
        if cond_drop_prob > 0:
            keep_mask = torch.rand(x.shape[0], device=x.device).less(1 - cond_drop_prob)
            cond_null = torch.zeros_like(cond)
            cond = torch.where(
                keep_mask.reshape(-1, *(1,) * (x.ndim - 1)),
                cond,
                cond_null
            )
        return self.torch_model(x, t, cond)

    def forward_with_cond_scale(self, *args, cond_scale=1., rescaled_phi=0., **kwargs):
        logits = self.forward(*args, cond_drop_prob=0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def training_step(self, batch, batch_idx):
        data_dict, info = batch
        ecg, ecg_noisy = data_dict['ecg'], data_dict['ecg_noisy']
        ecg_inputs, ecg_noisy_inputs = self.preprocess_inputs(ecg, ecg_noisy)
        t = torch.randint(0, self.n_diffusion_steps, (ecg.shape[0],), device=self.device).long()
        loss = self.p_losses(ecg_inputs, t, ecg_noisy_inputs)
        step_output = {
            'metrics': {
                'loss': loss,
            },
            'info': {k: tensor_to_array(v) for k, v in info.items()},
        }
        self.log('loss', loss, prog_bar=True, sync_dist=True, batch_size=ecg.shape[0])

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        return step_output

    def validation_step(self, batch, batch_idx):
        data_dict, info = batch
        ecg, ecg_noisy = data_dict['ecg'], data_dict['ecg_noisy']
        ecg_inputs, ecg_noisy_inputs = self.preprocess_inputs(ecg, ecg_noisy)
        ecg_denoised = self.sample(ecg_noisy_inputs.shape, self.clip_x_start, ecg_noisy_inputs, self.cond_scale, self.rescaled_phi)
        ecg_denoised = self.postprocess_outputs(ecg, ecg_denoised)
        step_output = {
            'ecg': (ecg_denoised.detach().cpu(), ecg.detach().cpu()),  # MetricsTool
            'ecg_original': tensor_to_array(ecg),  # ShowSamples, PredictionRecorder
            'ecg_noisy': tensor_to_array(ecg_noisy),  # ShowSamples, PredictionRecorder
            'ecg_denoised': tensor_to_array(ecg_denoised),  # ShowSamples, PredictionRecorder
            'info': {k: tensor_to_array(v) for k, v in info.items()},
        }
        return step_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.torch_model.parameters(), lr=2e-4, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50], gamma=0.5)
        return [optimizer], [lr_scheduler]
