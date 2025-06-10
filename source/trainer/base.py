# -*- coding: utf-8 -*-
import os
import sys
import lightning.pytorch as pl
import torch

from source.data.base import ZeroMean, Normalize, Scale

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AbstractTrainer(pl.LightningModule):
    def __init__(self, torch_model: torch.nn.Module, preprocess: bool):
        super().__init__()
        self.automatic_optimization = False
        self.torch_model = torch_model
        self.preprocessors = [
            ZeroMean(),
            Normalize(),
            Scale(2, -1),
        ] if preprocess else []

    def preprocess_inputs(self, ecg, ecg_noisy):
        for p in self.preprocessors:
            # 不能fit ecg，因为真实情况下不知道干净信号
            p.fit(ecg_noisy)
            ecg = p.transform(ecg)
            ecg_noisy = p.transform(ecg_noisy)
        return ecg, ecg_noisy

    def postprocess_outputs(self, ecg, ecg_denoised, baseline_align=True):
        for p in reversed(self.preprocessors):
            ecg_denoised = p.reverse(ecg_denoised)
        # 基线对齐
        if baseline_align:
            ecg_denoised = ecg_denoised - torch.mean(ecg_denoised, 1, True) + torch.mean(ecg, 1, True)
        return ecg_denoised

    def forward(self, *args, **kwargs):
        return self.torch_model(*args, **kwargs)

    def on_train_epoch_end(self) -> None:
        lrs = self.lr_schedulers()
        if lrs is not None:
            lrs.step()
            self.log('lr', lrs.get_last_lr()[0], prog_bar=True, sync_dist=True)

    def save_torch_model(self, checkpoint_path, *args, **kwargs):
        torch.save(self.torch_model.state_dict(), checkpoint_path)
        print(f"Saved: {checkpoint_path}")

    def load_torch_model(self, checkpoint_path, *args, **kwargs):
        self.torch_model.load_state_dict(torch.load(checkpoint_path, "cpu"))


def main():
    x = torch.arange(10).reshape(1, 10, 1).float()
    y = x + 1
    t = AbstractTrainer(None)
    x1, y1 = t.preprocess_inputs(x, y)
    x0 = t.postprocess_outputs(x, x1)
    y0 = t.postprocess_outputs(x, y1)
    print(x, y)
    print(x1, y1)
    print(x0, y0)


if __name__ == "__main__":
    main()
