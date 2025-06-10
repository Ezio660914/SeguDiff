# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from source.data.mit_bih import NSTDB

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ECGNoise:
    noise_name = ('ma', 'em', 'bw')

    @staticmethod
    def get_nstdb_data_df_dict(nstdb_dir, start=0, end=-1):
        data_df_dict = {}
        for name in ECGNoise.noise_name:
            file = NSTDB(os.path.join(nstdb_dir, name))
            data = file.get_ecg_data().drop(columns=['time', 'label'])
            data = data.iloc[start:end]
            data_df_dict[name] = data
        return data_df_dict

    def __init__(
            self,
            data_df_dict,
            segment_len=512,
            hybrid_weights=(1, 1, 1),
            loop=False,
            used_channels=(0, 1)
    ):
        self.data_df_dict = data_df_dict
        self.segment_len = segment_len
        self.hybrid_weights = hybrid_weights
        self.loop = loop
        self.hybrid_noise = None
        self.used_channels = used_channels
        self.set_hybrid_weights(hybrid_weights)

    def set_hybrid_weights(self, hybrid_weights):
        self.hybrid_noise = np.stack([df[['data_1', 'data_2']].values for df in self.data_df_dict.values()], 0)
        self.hybrid_noise = np.average(self.hybrid_noise, 0, hybrid_weights)

    def __len__(self):
        return self.hybrid_noise.shape[0] - self.segment_len + 1

    def __getitem__(self, start):
        if self.loop:
            start = start % self.hybrid_noise.shape[0]
        if self.loop and start + self.segment_len > self.hybrid_noise.shape[0]:
            noise = np.concatenate([self.hybrid_noise[start:], self.hybrid_noise[:start + self.segment_len - self.hybrid_noise.shape[0]]], 0)
        else:
            noise = self.hybrid_noise[start:start + self.segment_len]
        noise = noise[:, self.used_channels]
        return noise
