# -*- coding: utf-8 -*-
import os
import sys

import torch

from source.experiments.base.pipeline import ExperimentPipeline
from source.experiments.ecg_denoise.pipeline import LUDBDataPipeline, SeguDiffPipeline, DenoiseSegmentationResultSubmission
from source.experiments.ecg_denoise.config import ECGDenoiseSegmentationConfig

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    config = ECGDenoiseSegmentationConfig()
    data_pipeline = LUDBDataPipeline(config)
    data_pipeline.prepare_data()
    model_pipeline = SeguDiffPipeline(config)
    solution = ExperimentPipeline(config, data_pipeline, model_pipeline)
    solution.configure_hyperparams()
    solution.fit()
    result_pipeline = DenoiseSegmentationResultSubmission(config, data_pipeline)
    result_pipeline.merge_results_2()


if __name__ == "__main__":
    main()
