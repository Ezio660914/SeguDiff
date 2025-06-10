# SeguDiff

This is the official implementation of SeguDiff for ECG denoise

## ECG Datasets

This study utilizes three publicly available ECG datasets from [PhysioNet](https://physionet.org/), covering standard rhythm recordings, real-world noise types, and detailed waveform annotations. The datasets used are as follows:

| Name                                                       | Description                                                                                                                           | Link                                           |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Lobachevsky University Electrocardiography Database (LUDB) | 200 10-second 12-lead ECG with complexes and waves annotated, at 500Hz                                                                | https://www.physionet.org/content/ludb/1.0.1/  |
| MIT-BIH Arrhythmia Database (MITDB)                        | 48 half-hour excerpts of two-channel ambulatory ECG, at 360Hz                                                                         | https://www.physionet.org/content/mitdb/1.0.0/ |
| MIT-BIH Noise Stress Test Database (NSTDB)                 | half-hour recordings of ECG noise, including baseline wander (bw), muscle artifact (ma), and electrode motion artifact (em), at 360Hz | https://physionet.org/content/nstdb/1.0.0/     |

All these datasets have been uploaded in the "data" folder.

## Environment Setup

1. Please create a virtual environment with `python 3.10`
2. Install main packages `pytorch 2.5.1` and `pytorch lightning 2.5.0`
3. Install other essential packages listed in the `requirements.txt`

## Run Experiment

To train our SeguDiff model on the LUDB dataset, run `train_segudiff.py` in the folder `source/experiments/ecg_denoise`.
