# -*- coding: utf-8 -*-
import os
import sys

import math
import torch
from einops import rearrange, reduce
from torch.nn import functional as F
from tqdm import tqdm
from lion_pytorch import Lion
from source.trainer.rddm import RDDM
from ml_utils.utils.register import Register
from ml_utils.utils.array_tools import extract, tensor_to_array

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@Register(group_name="trainer", func_name="RDDMSegmentation")
class RDDMSegmentation(RDDM):
    def __init__(self, denoise_model, num_classes, ce_loss_weight, segmentation_model=None, *args, **kwargs):
        super().__init__(denoise_model, *args, objective="pred_res_noise", **kwargs)
        self.segmentation_model = segmentation_model
        if self.segmentation_model is not None:
            self.segmentation_model.eval()
        self.num_classes = num_classes
        self.ce_loss_weight = ce_loss_weight

    def p_mean_variance(self, x, t, cond, *args, **kwargs):
        model_output, pred_res, pred_noise, x_start_pred, mask_denoised = self.forward(x, t, cond)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(pred_res, x_start_pred, x, t)
        return model_mean, posterior_variance, posterior_log_variance, x_start_pred, mask_denoised

    def p_sample(self, x, t: int, cond, *args, **kwargs):
        batched_times = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        model_mean, posterior_variance, posterior_log_variance, x_start_pred, mask_denoised = self.p_mean_variance(x, batched_times, cond)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        x_t_prev = model_mean + (0.5 * posterior_log_variance).exp() * noise
        return x_t_prev, x_start_pred, mask_denoised

    def p_sample_loop(self, x_input=None, cond=None, return_sequence=False, *args, **kwargs):
        x_t = x_input + math.sqrt(self.sum_scale) * torch.randn_like(x_input)
        x_list = [x_t] if return_sequence else None
        mask_denoised = None
        for t in tqdm(reversed(range(0, self.n_diffusion_steps)), desc='ddpm sampling', total=self.n_diffusion_steps):
            x_t, x_start, mask_denoised = self.p_sample(x_t, t, cond)
            if return_sequence:
                x_list.append(x_t)
        return (x_list if return_sequence else x_t, mask_denoised)

    def fast_loop(self, x_input=None, cond=None, n_steps=3, *args, **kwargs):
        x_t = x_input + math.sqrt(self.sum_scale) * torch.randn_like(x_input)
        x_start = x_t
        mask_denoised = None
        for t in reversed(range(self.n_diffusion_steps - n_steps, self.n_diffusion_steps)):
            x_t, x_start, mask_denoised = self.p_sample(x_t, t, cond)
        return x_start, mask_denoised

    def ddim_sample(self, x_input=None, cond=None, n_sampling_steps=None, ddim_sampling_eta=None, return_sequence=False, *args, **kwargs):
        n_sampling_steps = self.n_sampling_steps if n_sampling_steps is None else n_sampling_steps
        ddim_sampling_eta = self.ddim_sampling_eta if ddim_sampling_eta is None else ddim_sampling_eta
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.n_diffusion_steps - 1,
                               steps=n_sampling_steps + 1)
        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        x_t = x_input + math.sqrt(self.sum_scale) * torch.randn_like(x_input)
        x_list = [x_t] if return_sequence else None
        mask_denoised = None
        for time, time_next in tqdm(time_pairs, desc='ddim sampling'):
            batched_times = torch.full((x_t.shape[0],), time, device=self.device, dtype=torch.long)
            model_output, pred_res, pred_noise, x_start_pred, mask_denoised = self.forward(x_t, batched_times, cond)

            if time_next < 0:
                x_t = x_start_pred
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
                  (1 - sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * x_start_pred + \
                  (alpha_cumsum_next - alpha_cumsum * sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum) * pred_res + \
                  sigma2.sqrt() * noise
            if return_sequence:
                x_list.append(x_t)
        return (x_list if return_sequence else x_t, mask_denoised)

    def p_losses(self, x_start, t, x_input, cond=None, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        x_res = x_input - x_start

        # noise sample
        x = self.q_sample(x_start, x_res, t, noise)

        # predict and take gradient step

        model_output, pred_res, pred_noise, x_start_pred, mask_denoised = self.forward(x, t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_res':
            target = x_res
        elif self.objective == 'pred_res_noise':
            target = torch.cat([x_res, noise], 2)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return model_output, target, pred_res, pred_noise, x_start_pred, mask_denoised

    def forward_self_reg(self, x_start, t, cond, noise):
        x = self.q_sample(x_start, 0, t, noise)
        return self.forward(x, t, cond)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def forward(self, x, t, cond, *args, **kwargs):
        x = x.clone().requires_grad_()
        t = t.clone()
        cond = cond.clone()
        model_output = self.torch_model(x, t, cond)
        pred_res, mask_denoised = torch.tensor_split(model_output, [model_output.shape[2] - self.num_classes], 2)
        pred_noise = torch.autograd.grad(torch.logsumexp(mask_denoised, dim=2).sum(), [x], create_graph=self.training, retain_graph=self.training)[0]
        pred_noise = -pred_noise
        model_output = torch.cat([pred_res, pred_noise], dim=2)
        x_start_pred = self.predict_start_from_res_noise(x, t, pred_res, pred_noise)

        return model_output, pred_res, pred_noise, x_start_pred, mask_denoised

    def pre_segmentation(self, ecg_noisy_inputs):
        with torch.no_grad():
            mask_logits = self.segmentation_model(ecg_noisy_inputs)
        return mask_logits

    def training_step(self, batch, batch_idx):
        data_dict, info = batch
        ecg_inputs, ecg_noisy_inputs = self.preprocess_inputs(data_dict['ecg'], data_dict['ecg_noisy'])
        t = torch.randint(0, self.n_diffusion_steps, (data_dict['ecg'].shape[0],), device=self.device).long()
        noise = torch.randn_like(ecg_inputs)

        if self.segmentation_model is None:
            cond = ecg_noisy_inputs
        else:
            mask_logits = self.pre_segmentation(ecg_noisy_inputs)
            cond = torch.cat([ecg_noisy_inputs, mask_logits], dim=2)

        t_0 = torch.zeros((data_dict['ecg'].shape[0],), dtype=torch.long, device=self.device)
        model_output_0 = self.torch_model(ecg_inputs, t_0, cond)
        _, mask_denoised_0 = torch.tensor_split(model_output_0, [model_output_0.shape[2] - self.num_classes], 2)

        model_out, target, pred_res, pred_noise, x_start_pred, mask_denoised = self.p_losses(ecg_inputs, t, ecg_noisy_inputs, cond, noise)

        step_output = {
            'metrics': {},
            'info': {k: tensor_to_array(v) for k, v in info.items()}
        }
        loss = 0

        p_loss = F.mse_loss(model_out, target, reduction="none")
        p_loss = reduce(p_loss, "b ... -> b", "mean")
        p_loss = p_loss * extract(self.loss_weight, t, p_loss.shape)
        p_loss = p_loss.mean()
        step_output['metrics']['p_loss'] = p_loss
        loss = loss + p_loss

        mask_label = data_dict['pqrst']
        mask_label_flat = rearrange(mask_label, 'b l c -> (b l) c').long()
        if mask_label_flat.shape[1] > 1:
            mask_label_flat = mask_label_flat.argmax(dim=1)
        else:
            mask_label_flat = mask_label_flat.squeeze(1)
        mask_denoised_flat = rearrange(mask_denoised, 'b l c -> (b l) c')
        ce_loss = F.cross_entropy(mask_denoised_flat, mask_label_flat, reduction='none')
        # ce_loss = rearrange(ce_loss, '(b l) -> b l', b=mask_label.shape[0])
        # ce_loss = reduce(ce_loss, "b ... -> b", "mean")
        # ce_loss = ce_loss * extract(self.one_minus_alphas_cumsum, t, ce_loss.shape)
        ce_loss = self.ce_loss_weight * ce_loss.mean()

        mask_denoised_0_flat = rearrange(mask_denoised_0, 'b l c -> (b l) c')
        ce_loss_0 = F.cross_entropy(mask_denoised_0_flat, mask_label_flat, reduction='none')
        ce_loss_0 = self.ce_loss_weight * ce_loss_0.mean()
        step_output['metrics']['ce_loss'] = ce_loss + ce_loss_0
        loss = loss + ce_loss + ce_loss_0

        step_output['metrics']['loss'] = loss
        self.log('loss', loss, prog_bar=True, sync_dist=True, batch_size=data_dict['ecg'].shape[0])

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        lrs = self.lr_schedulers()
        if lrs is not None and self.global_step > 0 and self.global_step % 100 == 0:
            lrs.step(self.global_step / 1000)
            self.log('lr', lrs.get_last_lr()[0], prog_bar=True, sync_dist=True)
        return step_output

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        data_dict, info = batch
        ecg_inputs, ecg_noisy_inputs = self.preprocess_inputs(data_dict['ecg'], data_dict['ecg_noisy'])
        if self.segmentation_model is None:
            cond = ecg_noisy_inputs
        else:
            mask_logits = self.pre_segmentation(ecg_noisy_inputs)
            cond = torch.cat([ecg_noisy_inputs, mask_logits], dim=2)
        ecg_denoised, mask_denoised = self.fast_loop(ecg_noisy_inputs, cond, 3)
        ecg_denoised = self.postprocess_outputs(data_dict['ecg'], ecg_denoised)

        # 分割结果
        if 'pqrst' in data_dict:
            mask_label = data_dict['pqrst']
        else:
            mask_label = torch.zeros_like(mask_denoised)[:, :, [0]]
        mask_label_flat = rearrange(mask_label, 'b l c -> (b l) c').long()
        if mask_label_flat.shape[1] > 1:
            mask_label_flat = mask_label_flat.argmax(dim=1)
        else:
            mask_label_flat = mask_label_flat.squeeze(1)
        mask_denoised = torch.softmax(mask_denoised, dim=2)
        mask_denoised_flat = rearrange(mask_denoised, 'b l c -> (b l) c')

        loss = F.mse_loss(ecg_denoised, data_dict['ecg']).sqrt()
        step_output = {
            'metrics': {},
            'pqrst': (mask_denoised_flat.detach().cpu(), mask_label_flat.detach().cpu()),  # MetricsTool
            'denoise': (ecg_denoised.detach().cpu(), data_dict['ecg'].detach().cpu()),  # MetricsTool
            'ecg': tensor_to_array(data_dict['ecg']),  # ShowSamples, PredictionRecorder
            'ecg_noisy': tensor_to_array(data_dict['ecg_noisy']),  # ShowSamples, PredictionRecorder
            'ecg_denoised': tensor_to_array(ecg_denoised),  # ShowSamples, PredictionRecorder
            'preds': tensor_to_array(mask_denoised),
            'labels': tensor_to_array(mask_label),
            'info': {k: tensor_to_array(v) for k, v in info.items()},
        }
        step_output['metrics']['val_loss'] = loss
        return step_output

    def configure_optimizers(self):
        optimizer = Lion(self.torch_model.parameters(), lr=1e-5, weight_decay=1e-4)
        return optimizer
