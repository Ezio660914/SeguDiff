# -*- coding: utf-8 -*-
import glob
import os
import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from torch.utils import data as td
from torchinfo import summary

from ml_utils.data.base import DataModuleBase
from ml_utils.experiments.base.data_pipeline import IResultSubmission, IDataPreparation
from ml_utils.utils.path_tools import get_stem
from source.data.base import SampleWisePreprocessDataset
from source.data.ludb import LUDB
from source.data.mit_bih import MITBIHDatabase
from source.data.nstdb import ECGNoise
from source.experiments.base.pipeline import ModelPipeline
from source.experiments.ecg_denoise.result_analysis import file_metrics_denoise, overall_metrics_denoise, file_segment_metrics_denoise
from source.trainer.ecg_denoise_segmentation import RDDMSegmentation
from source.trainer.rddm import RDDM
from source.models.unet import UNet1d
from ml_utils.utils.register import Register

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class NSTDBDataPipeline(IDataPreparation):
    def __init__(self, config):
        super().__init__(config)
        self.noise_dataset_list = None

    def prepare_nstdb_data(self):
        nstdb_dir = os.path.join(self.config.data_dir, "mit-bih-noise-stress-test-database-1.0.0")
        self.noise_dataset_list = [[], []]
        if self.config.training:
            noisy_data_df_dict_train = ECGNoise.get_nstdb_data_df_dict(nstdb_dir, self.config.train_noise_start, self.config.train_noise_end)
            for hybrid_weights in self.config.train_hybrid_weights:
                dataset = ECGNoise(noisy_data_df_dict_train, self.config.window_size, hybrid_weights, False, self.config.train_noise_used_channels)
                self.noise_dataset_list[0].append(dataset)
        if self.config.validate:
            noisy_data_df_dict_val = ECGNoise.get_nstdb_data_df_dict(nstdb_dir, self.config.val_noise_start, self.config.val_noise_end)
            for hybrid_weights in self.config.val_hybrid_weights:
                dataset = ECGNoise(noisy_data_df_dict_val, self.config.window_size, hybrid_weights, False, self.config.val_noise_used_channels)
                self.noise_dataset_list[1].append(dataset)


class MITDBDataPipeline(IDataPreparation):
    def __init__(self, config):
        super().__init__(config)
        self.nstdb_data_pipeline = NSTDBDataPipeline(config)
        self.data_dir = os.path.join(self.config.data_dir, "mit-bih-arrhythmia-database-1.0.0")
        self.noise_label_dir = os.path.join(self.config.data_dir, "MITDB_noise_annotations") if self.config.use_noise_labels else None
        self.train_database = None
        self.val_database = None
        self.folds = None

    def prepare_data(self):
        self.nstdb_data_pipeline.prepare_nstdb_data()
        self.prepare_mitbih_data()

    def prepare_mitbih_data(self):
        file_names = [get_stem(f) for f in glob.iglob(os.path.join(self.data_dir, "*.dat"))]
        fold_config = os.path.join(self.data_dir, self.config.fold_config)
        if os.path.exists(fold_config):
            with open(fold_config) as f:
                train_file_names, val_file_names = yaml.safe_load(f)
        else:
            # train_file_names, val_file_names = map(lambda x: x.tolist(), split_array(np.asarray(file_names), self.config.train_ratio))
            # 仅验证有MLII的所有数据
            train_file_names = []
            val_file_names = list(set(file_names) - {"102", "104"})
            with open(fold_config, 'w') as f:
                yaml.safe_dump([train_file_names, val_file_names], f)
            print(f"fold config saved to {fold_config}")
        train_file_list = list(map(lambda x: os.path.join(self.data_dir, get_stem(x)), train_file_names))
        val_file_list = list(map(lambda x: os.path.join(self.data_dir, get_stem(x)), val_file_names))
        self.folds = [[train_file_list, val_file_list]]

    def prepare_paired_datasets(self, ecg_dataset, noise_dataset_list, snr_list, repeat_ecg=0):
        if self.config.dataset_prefilter:
            ecg_dataset = SampleWisePreprocessDataset.prefilter_ecg(ecg_dataset)
        paired_dataset = []
        for snr, noise_dataset in product(snr_list, noise_dataset_list):
            ecg_noisy_dataset = SampleWisePreprocessDataset(ecg_dataset, noise_dataset, snr, self.config.dataset_preprocess, repeat_ecg)
            paired_dataset.append(ecg_noisy_dataset)
            print(f"create paired dataset: snr={snr}, hybrid_weights={noise_dataset.hybrid_weights}")
        paired_dataset = td.ConcatDataset(paired_dataset)
        return paired_dataset

    def configure_fold_datamodule(self, fold_id, *args, **kwargs):
        train_file_list, val_file_list = self.folds[fold_id]
        if self.config.training:
            print("train files", train_file_list)
            self.train_database = MITBIHDatabase(train_file_list, self.config.window_size, "MITDB", used_channels=self.config.used_channels, noise_label_dir=self.noise_label_dir,
                                                 window_sep=self.config.window_sep)
            self.train_database.preprocess_all()
            train_dataset = self.train_database.get_dataset()
            train_dataset = self.prepare_paired_datasets(train_dataset, self.nstdb_data_pipeline.noise_dataset_list[0], self.config.train_snr, 0)
        else:
            train_dataset = None
        if self.config.validate:
            print("val files", val_file_list)
            self.val_database = MITBIHDatabase(val_file_list, self.config.window_size, "MITDB", used_channels=self.config.used_channels)
            self.val_database.preprocess_all()
            val_dataset = self.val_database.get_dataset()
            val_dataset = self.prepare_paired_datasets(val_dataset, self.nstdb_data_pipeline.noise_dataset_list[1], self.config.val_snr, 0)
        else:
            val_dataset = None
        datamodule = DataModuleBase(train_dataset, val_dataset, train_batch_size=self.config.train_batch_size, val_batch_size=self.config.val_batch_size)
        return datamodule


class LUDBDataPipeline(IDataPreparation):
    def __init__(self, config):
        super().__init__(config)
        self.nstdb_data_pipeline = NSTDBDataPipeline(config)
        self.data_dir = os.path.join(self.config.data_dir, "lobachevsky-university-electrocardiography-database-1.0.1")
        self.ludb_csv_file = os.path.join(self.data_dir, "ludb.csv")
        self.train_database = None
        self.val_database = None

    def prepare_data(self):
        self.nstdb_data_pipeline.prepare_nstdb_data()
        self.prepare_ludb_data()

    def prepare_ludb_data(self):
        file_list = list(map(lambda x: os.path.splitext(x)[0], glob.iglob(os.path.join(self.data_dir, "data", "*.dat"))))
        self.database = LUDB(file_list, self.config.window_size, self.config.window_sep, self.config.used_channels, 360, self.ludb_csv_file)
        fold_config = os.path.join(self.config.yaml_dir, self.config.fold_config)
        if os.path.exists(fold_config):
            self.database.load_folds(fold_config)
        else:
            self.database.make_folds(self.config.k_folds)
            self.database.save_folds(fold_config)
            print(f"fold config saved to {fold_config}")

    def prepare_paired_datasets(self, ecg_dataset, noise_dataset_list, snr_list, repeat_ecg=0):
        if self.config.dataset_prefilter:
            ecg_dataset = SampleWisePreprocessDataset.prefilter_ecg(ecg_dataset)
        paired_dataset = []
        for snr, noise_dataset in product(snr_list, noise_dataset_list):
            ecg_noisy_dataset = SampleWisePreprocessDataset(ecg_dataset, noise_dataset, snr, self.config.dataset_preprocess, repeat_ecg)
            paired_dataset.append(ecg_noisy_dataset)
            print(f"create paired dataset: snr={snr}, hybrid_weights={noise_dataset.hybrid_weights}")
        paired_dataset = td.ConcatDataset(paired_dataset)
        return paired_dataset

    def configure_fold_datamodule(self, fold_id, *args, **kwargs):
        train_file_names, val_file_names = self.database.folds[fold_id]
        if self.config.training:
            train_file_list = list(map(lambda x: os.path.join(self.data_dir, 'data', str(x)), train_file_names))
            print("train file names", train_file_names)
            self.train_database = LUDB(train_file_list, self.config.window_size, self.config.window_sep, self.config.used_channels, 360, self.ludb_csv_file)
            self.train_database.preprocess_all()
            train_dataset = self.train_database.get_dataset()
            train_dataset = self.prepare_paired_datasets(train_dataset, self.nstdb_data_pipeline.noise_dataset_list[0], self.config.train_snr, 650000 // 2 // 3600 - 1)
        else:
            train_dataset = None
        if self.config.validate:
            val_file_list = list(map(lambda x: os.path.join(self.data_dir, 'data', str(x)), val_file_names))
            print("val file names", val_file_names)
            # 测试时使用无重叠窗口
            self.val_database = LUDB(val_file_list, self.config.window_size, None, self.config.used_channels, 360, self.ludb_csv_file)
            self.val_database.preprocess_all()
            val_dataset = self.val_database.get_dataset()
            val_dataset = self.prepare_paired_datasets(val_dataset, self.nstdb_data_pipeline.noise_dataset_list[1], self.config.val_snr, 650000 // 2 // 3600 - 1)
        else:
            val_dataset = None
        datamodule = DataModuleBase(train_dataset, val_dataset, train_batch_size=self.config.train_batch_size, val_batch_size=self.config.val_batch_size)
        return datamodule


class DenoiseSegmentationResultSubmission(IResultSubmission):
    def __init__(self, config, data_pipeline):
        super().__init__(config)
        self.data_pipeline = data_pipeline

    def merge_results(self):
        os.makedirs(self.config.result_save_dir, exist_ok=True)
        data_df_dict = self.data_pipeline.val_database.data_df_dict.copy()

        with open(self.config.log_file, 'rb') as f:
            step_output = pickle.load(f)
        ecg_original, ecg_noisy, ecg_denoised, segment_preds, segment_labels, batch_info = next(iter(step_output))
        B, L, C = ecg_original.shape
        original_col_names = [f'ecg_original_{i}' for i in range(C)]
        noisy_col_names = [f'ecg_noisy_{i}' for i in range(C)]
        denoised_col_names = [f'ecg_denoised_{i}' for i in range(C)]

        segmentation_col_names = list(map(lambda x: '_'.join(x), product(['preds', 'labels'], ['BG', 'P', 'QRS', 'T'])))
        col_names = original_col_names + noisy_col_names + denoised_col_names + segmentation_col_names
        for ecg_original, ecg_noisy, ecg_denoised, segment_preds, segment_labels, batch_info in tqdm.tqdm(step_output, 'merge results'):
            B, *_ = ecg_original.shape
            for i in range(B):
                file_name = batch_info['file_name'][i]
                if col_names[0] not in data_df_dict[file_name].columns:
                    data_df_dict[file_name][col_names] = np.nan
                start = batch_info['index'][i]
                end = start + batch_info['window_size'][i]
                # "end" needs to minus 1 if using .loc
                data_df_dict[file_name].loc[start:end - 1, col_names] = np.concatenate([ecg_original[i], ecg_noisy[i], ecg_denoised[i], segment_preds[i], segment_labels[i]], 1)

        for file_name, data_df in tqdm.tqdm(data_df_dict.items(), 'save csv', len(data_df_dict)):
            data_df.to_csv(os.path.join(self.config.result_save_dir, f"{file_name}_result.csv"), index=False)

        # output metrics
        result_files = glob.glob(os.path.join(self.config.result_save_dir, '*result.csv'))
        file_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names)
        overall_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names)
        file_segment_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names, self.config.window_size)

    def merge_results_2(self):
        os.makedirs(self.config.result_save_dir, exist_ok=True)
        data_df_dict = {}

        with open(self.config.log_file, 'rb') as f:
            step_output = pickle.load(f)
        ecg_original, ecg_noisy, ecg_denoised, segment_preds, segment_labels, batch_info = next(iter(step_output))
        B, L, C = ecg_original.shape
        original_col_names = [f'ecg_original_{i}' for i in range(C)]
        noisy_col_names = [f'ecg_noisy_{i}' for i in range(C)]
        denoised_col_names = [f'ecg_denoised_{i}' for i in range(C)]
        if segment_labels.shape[-1] == 1:
            segmentation_col_names = list(map(lambda x: '_'.join(x), product(['preds'], ['BG', 'P', 'QRS', 'T']))) + ['labels']
        elif segment_labels.shape[-1] == 4:
            segmentation_col_names = list(map(lambda x: '_'.join(x), product(['preds', 'labels'], ['BG', 'P', 'QRS', 'T'])))
        else:
            raise RuntimeError
        col_names = original_col_names + noisy_col_names + denoised_col_names + segmentation_col_names
        for ecg_original, ecg_noisy, ecg_denoised, segment_preds, segment_labels, batch_info in tqdm.tqdm(step_output, 'merge results'):
            B, *_ = ecg_original.shape
            for i in range(B):
                file_name = batch_info['file_name'][i]
                if file_name not in data_df_dict:
                    data_df_dict[file_name] = []
                data_df_dict[file_name].append(np.concatenate([ecg_original[i], ecg_noisy[i], ecg_denoised[i], segment_preds[i], segment_labels[i]], 1))

        for file_name, data_df in tqdm.tqdm(data_df_dict.items(), 'save csv', len(data_df_dict)):
            data_df = np.concatenate(data_df, axis=0)
            data_df = pd.DataFrame(data_df, columns=col_names)
            data_df.to_csv(os.path.join(self.config.result_save_dir, f"{file_name}_result.csv"), index=False)

        # output metrics
        result_files = glob.glob(os.path.join(self.config.result_save_dir, '*result.csv'))
        file_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names)
        overall_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names)
        file_segment_metrics_denoise(result_files, self.config.result_save_dir, denoised_col_names, original_col_names, self.config.window_size)


class SeguDiffPipeline(ModelPipeline):
    def print_summary(self, model, *args, **kwargs):
        model_args = self.yaml_config['models']['denoise_model']['args']
        x = torch.randn(1, self.config.window_size, model_args['in_channels'])
        t = torch.zeros(1, dtype=torch.int64)
        c = torch.randn(1, self.config.window_size, model_args['context_dim'])
        summary(model, input_data=(x, t, c), depth=3, device='cpu')
