# -*- coding: utf-8 -*-
import glob
import os
import sys
from typing import Literal

import numpy as np
import wfdb as w
from wfdb import processing as wp
import pandas as pd
from source.data.base import AbstractDatabase, SegmentIterator, DatasetName
from ml_utils.utils.array_tools import get_windowed_data
from ml_utils.utils.mp import async_run
from ml_utils.utils.path_tools import get_stem

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MitBihFile:
    def __init__(self, record_name, ecg_sample_rate=None, noise_label_file=None):
        self.record_name = record_name
        self.ecg_sample_rate = ecg_sample_rate
        if os.path.exists(self.record_name + ".atr"):
            self.annotation = w.rdann(self.record_name, 'atr')
        else:
            self.annotation = None
        self.noise_label_file = noise_label_file

    def get_ecg_data(self):
        """
        :return: ecg data [length, time + 2 channels + label + r peak label]
        """
        record, info = w.rdsamp(self.record_name)
        if isinstance(self.ecg_sample_rate, int):
            record, annotation = wp.resample_multichan(record, self.annotation, self.annotation.fs, self.ecg_sample_rate)
        else:
            annotation = self.annotation
        label = np.zeros(record.shape[0])
        start = 0
        af_begin = False
        for idx, label_name in zip(annotation.sample, annotation.aux_note):
            if "AFIB" in label_name:
                if not af_begin:
                    start = idx
                    af_begin = True
            elif len(label_name) > 0:
                if af_begin:
                    label[start:idx] = 1
                    af_begin = False
        # process the end of the segment
        if af_begin:
            label[start:] = 1
        # noise label
        for idx, label_name in zip(annotation.sample, annotation.symbol):
            if label_name in ['|', '~']:
                label[idx] = -1
        # 使用标注软件标记的噪声
        if self.noise_label_file is not None and os.path.exists(self.noise_label_file):
            noise_label_df = pd.read_csv(self.noise_label_file)
            for row in noise_label_df.itertuples(index=False):
                start = round(row.start * annotation.fs)
                end = round(row.end * annotation.fs) + 1
                assert start <= end
                label[start:end] = -1
        time = np.arange(record.shape[0]) / annotation.fs
        r_peak_position = self.get_r_peak_position()
        r_peak_position_label = np.zeros(record.shape[0])
        r_peak_position_label[r_peak_position] = 1
        data = np.c_[time, record, label, r_peak_position_label]
        columns = ['time', *MITBIHDatabase.default_channels, 'label', 'r_peak']
        # 心拍类型标签
        df = self.get_heartbeat_labels()
        if df is not None:
            hearbeat_label = np.zeros(data.shape[0])
            hearbeat_label[df['sample'].values.astype(int)] = df['heartbeat_label'].values
            data = np.concatenate([data, hearbeat_label.reshape(-1, 1)], axis=1)
            columns.append('heartbeat_label')
        data = pd.DataFrame(data, columns=columns)
        return data

    def get_r_peak_position(self):
        raise NotImplementedError

    def get_heartbeat_labels(self):
        return None

    def get_rri_data(self):
        """
        :return: rri data [length, time + rr interval + label]
        """
        r_peak_position = self.get_r_peak_position()
        label = np.zeros_like(r_peak_position, float)
        start = 0
        af_begin = False
        for idx, label_name in zip(self.annotation.sample, self.annotation.aux_note):
            if "AFIB" in label_name:
                if not af_begin:
                    start = idx
                    af_begin = True
            elif len(label_name) > 0:
                if af_begin:
                    label[(r_peak_position >= start) & (r_peak_position <= idx)] = 1.
                    af_begin = False
        # process the end of the segment
        if af_begin:
            label[r_peak_position >= start] = 1
        # noise label
        for idx, label_name in zip(self.annotation.sample, self.annotation.symbol):
            if label_name in ['|', '~']:
                label[np.argmin(r_peak_position - idx)] = -1
        rri = np.diff(r_peak_position) / self.annotation.fs * 1000
        label = label[:-1]
        time = r_peak_position[:-1] / self.annotation.fs
        data = np.c_[time, rri, label]
        return data


class AFDB(MitBihFile):
    def get_r_peak_position(self):
        if os.path.exists(self.record_name + '.qrsc'):
            qrs = w.rdann(self.record_name, 'qrsc')
        elif os.path.exists(self.record_name + '.qrs'):
            qrs = w.rdann(self.record_name, 'qrs')
        else:
            raise FileNotFoundError
        return qrs.sample


class MITDB(MitBihFile):
    BEAT_ANNOTATIONS = list("NLRBAaJSVrFejnE/fQ?")
    BEAT_LABEL_VALUE = list("0005111125300524445")  # BEAT_ANNOTATIONS每个对应的AAMI label, 0=N, 1=S, 2=V, 3=F, 4=Q, 5=其他

    def get_r_peak_position(self):
        df = pd.DataFrame({"sample": self.annotation.sample, "symbol": self.annotation.symbol})
        r_peak_position = df.loc[df['symbol'].isin(MITDB.BEAT_ANNOTATIONS), "sample"].to_numpy()
        return r_peak_position

    def get_heartbeat_labels(self):
        df = pd.DataFrame({"sample": self.annotation.sample, "symbol": self.annotation.symbol})
        df = df[df['symbol'].isin(MITDB.BEAT_ANNOTATIONS)]
        label_map = {annot: int(label) for annot, label in zip(self.BEAT_ANNOTATIONS, self.BEAT_LABEL_VALUE)}
        df["heartbeat_label"] = df['symbol'].map(label_map)
        return df


class NSTDB(MitBihFile):
    def get_ecg_data(self):
        """
        :return: [length, time + 2 channels + label]
        """
        record, info = w.rdsamp(self.record_name)
        time = np.arange(record.shape[0]) / info['fs']
        label = -np.ones(record.shape[0])
        data = pd.DataFrame(np.c_[time, record, label], columns=['time', *MITBIHDatabase.default_channels, 'label'])
        return data


class MITBIHDatabase(AbstractDatabase):
    default_channels = ['data_1', 'data_2']

    def __init__(self, file_list, window_size, mode: Literal["AFDB", "MITDB", "NSTDB"], ecg_sample_rate=None, used_channels=None, noise_label_dir=None, window_sep=None):
        super().__init__(file_list, window_size, window_sep)
        self.used_channels = used_channels if used_channels is not None else self.default_channels
        self.ecg_sample_rate = ecg_sample_rate
        self.mode = mode
        self.noise_label_dir = noise_label_dir

    def preprocess_all(self):
        def _preprocess_file(file_name):
            if self.mode == 'AFDB':
                cls = AFDB
            elif self.mode == 'MITDB':
                cls = MITDB
            elif self.mode == 'NSTDB':
                cls = NSTDB
            else:
                raise ValueError
            if self.noise_label_dir is not None:
                noise_label_file = os.path.join(self.noise_label_dir, os.path.basename(file_name)) + ".labels.csv"
                if not os.path.exists(noise_label_file):
                    print(f"{noise_label_file} not found")
            else:
                noise_label_file = None
            dataset = cls(file_name, self.ecg_sample_rate, noise_label_file=noise_label_file)
            data_df = dataset.get_ecg_data()
            windowed_data, info = self.get_base_info(file_name, data_df.values)
            label_sum = np.sum(windowed_data[:, :, 4], 1)
            info['type'] = 'NAF'
            info.loc[label_sum > self.window_size // 2, 'type'] = 'PAF'
            # noisy
            info.loc[np.any(windowed_data[:, :, 4] == -1, axis=1), 'type'] = 'NOISE'
            return file_name, data_df, info

        result = async_run(_preprocess_file, self.file_list, lambda x: (x,), 'preprocess all')
        result = list(filter(lambda x: x is not None, result))
        file_name, data_df, info = list(zip(*result))
        self.data_df_dict = {get_stem(file_path): df for file_path, df in zip(file_name, data_df)}
        self.info = pd.concat(info, axis=0, ignore_index=True)
        return self.data_df_dict, self.info

    def get_dataset(self, fold_id=0, subset=None, balance_classes=False):
        """
        Args:
            fold_id: int
            subset: None for all, 0 for training set, 1 for validation set
            balance_classes: bool
        """
        if subset is None:
            info = self.info
        else:
            file_name = self.folds[fold_id][subset]
            info = self.info[self.info['file_name'].isin(file_name)]
        # 样本平衡
        if balance_classes:
            info = info.groupby('type').sample(info['type'].value_counts().min())
        print(f"Using fold {fold_id}")
        print(info['type'].value_counts())
        if np.any(info['type'] == "NOISE") and self.mode != "NSTDB":
            info = info[info['type'] != "NOISE"]
            print("NOISE Ignored")
        return SegmentIterator(self.data_df_dict, info, DatasetName.mit_bih, self.used_channels, [self.default_channels.index(x) for x in self.used_channels])
