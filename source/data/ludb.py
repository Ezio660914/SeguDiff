# -*- coding: utf-8 -*-
import os
import sys
import glob
from typing import Literal
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wfdb as w
from scipy import signal
from wfdb import processing as wp
import pandas as pd
from source.data.base import AbstractDatabase, SegmentIterator, DatasetName
from ml_utils.utils.array_tools import get_windowed_data
from ml_utils.utils.mp import async_run
from ml_utils.utils.path_tools import get_stem

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class LUDB(AbstractDatabase):
    ecg_sample_rate = 500
    default_channels = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
    pqrst_type_mapping = {0: "BG", 1: "P", 2: "QRS", 3: "T"}

    def __init__(self, file_list, window_size, window_sep=None, used_channels=None, fs_target=None, ludb_csv_file=None):
        super().__init__(file_list, window_size, window_sep)
        self.used_channels = used_channels if used_channels is not None else self.default_channels
        self.fs_target = fs_target
        self.ludb_csv = pd.read_csv(ludb_csv_file)
        self.ludb_csv['ID'] = self.ludb_csv['ID'].astype(str)
        self.ludb_csv = self.ludb_csv.set_index('ID')
        self.ludb_csv['label'] = self.ludb_csv['Rhythms'].apply(lambda x: 1 if ('Atrial fibrillation'.lower() in x.lower()) or ('Atrial flutter'.lower() in x.lower()) else 0)

    @staticmethod
    def ludb_file_to_dataframe(file_path, fs_target=None):
        record, info = w.rdsamp(file_path)
        if isinstance(fs_target, int) and fs_target != info['fs']:
            new_length = int(record.shape[0] * fs_target / info['fs'])
            record = signal.resample(record, new_length, axis=0)
        sig_name = info['sig_name']
        label_df_dict = {}
        for ld in sig_name:
            ann = w.rdann(file_path, extension=ld)
            sample = ann.sample
            if isinstance(fs_target, int):
                sample = wp.resample_ann(sample, info['fs'], fs_target)
            df_lead_ann = pd.DataFrame()
            symbols = np.array(ann.symbol)
            peak_inds = np.where(np.isin(symbols, ["p", "N", "t"]))[0]
            df_lead_ann["peak"] = sample[peak_inds]
            df_lead_ann["onset"] = np.nan
            df_lead_ann["offset"] = np.nan
            for i, row in df_lead_ann.iterrows():
                peak_idx = peak_inds[i]
                if peak_idx == 0:
                    df_lead_ann.loc[i, "onset"] = row["peak"]
                    if symbols[peak_idx + 1] == ")":
                        df_lead_ann.loc[i, "offset"] = sample[peak_idx + 1]
                    else:
                        df_lead_ann.loc[i, "offset"] = row["peak"]
                elif peak_idx == len(symbols) - 1:
                    df_lead_ann.loc[i, "offset"] = row["peak"]
                    if symbols[peak_idx - 1] == "(":
                        df_lead_ann.loc[i, "onset"] = sample[peak_idx - 1]
                    else:
                        df_lead_ann.loc[i, "onset"] = row["peak"]
                else:
                    if symbols[peak_idx - 1] == "(":
                        df_lead_ann.loc[i, "onset"] = sample[peak_idx - 1]
                    else:
                        df_lead_ann.loc[i, "onset"] = row["peak"]
                    if symbols[peak_idx + 1] == ")":
                        df_lead_ann.loc[i, "offset"] = sample[peak_idx + 1]
                    else:
                        df_lead_ann.loc[i, "offset"] = row["peak"]

            df_lead_ann.index = symbols[peak_inds]

            for c in ["peak", "onset", "offset"]:
                df_lead_ann[c] = df_lead_ann[c].values.astype(int)
            label_df_dict[ld] = df_lead_ann

        class_map = dict(p=1, N=2, t=3)
        data_df = pd.DataFrame(record, columns=LUDB.default_channels)
        for ld, df in label_df_dict.items():
            ch_label = np.zeros(record.shape[0], dtype=np.int32)
            for idx, row in df.iterrows():
                ch_label[int(row['onset']):int(row['offset'])] = class_map[idx]
            data_df[ld + "_label"] = ch_label
        return data_df

    def preprocess_all(self):
        def _preprocess_file(file_path):
            file_name = get_stem(file_path)
            data_df = self.ludb_file_to_dataframe(file_path, fs_target=self.fs_target)
            windowed_data, info = self.get_base_info(file_path, data_df.values)
            info['type'] = self.ludb_csv.loc[file_name, 'label']
            return file_name, data_df, info

        result = async_run(_preprocess_file, self.file_list, lambda x: (x,), 'preprocess all')
        result = list(filter(lambda x: x is not None, result))
        file_name, data_df, info = list(zip(*result))
        self.data_df_dict = {n: df for n, df in zip(file_name, data_df)}
        self.info = pd.concat(info, axis=0, ignore_index=True)
        return self.data_df_dict, self.info

    def get_dataset(self, fold_id=0, subset=None):
        """
        Args:
            fold_id: int
            subset: None for all, 0 for training set, 1 for validation set
        """
        if subset is None:
            info = self.info
        else:
            file_name = self.folds[fold_id][subset]
            info = self.info[self.info['file_name'].isin(file_name)]
        print(f"Using fold {fold_id}")
        print(info['type'].value_counts())
        return SegmentIterator(self.data_df_dict, info, DatasetName.ludb, self.used_channels, [self.default_channels.index(x) for x in self.used_channels])

    def make_folds(self, k_folds, *args, **kwargs):
        splitter = StratifiedKFold(n_splits=k_folds, shuffle=True)
        self.folds = []
        for train_idx, val_idx in splitter.split(self.ludb_csv.index, self.ludb_csv['label']):
            self.folds.append([self.ludb_csv.index[train_idx].tolist(), self.ludb_csv.index[val_idx].tolist()])
        return self.folds
