import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

import os.path as osp
import os
import numpy as np

import scipy.io as sio
import scipy.signal as ss
import pywt

import time

import matplotlib.pyplot as plt
plt.switch_backend('Agg')


DENOISE_DATA_DIRS = {
    'mitdb': "/home/workspace/mingchen/ECG_UDA/data/mitdb/",
    'incartdb': "/home/workspace/mingchen/ECG_UDA/data/incartdb",
    'svdb': "/home/workspace/mingchen/ECG_UDA/data/svdb"
}

SAMPLE_RATES = {
    'mitdb': 360,
    'incartdb': 257,
    'svdb': 128,
}


def load_dataset_to_memory(dataset_name):

    load_path = DENOISE_DATA_DIRS[dataset_name]
    records = os.listdir(load_path)

    data_dict = {}

    for record in records:
        data = sio.loadmat(osp.join(load_path, record))
        signal = data['signal']
        category = data['category'][0]
        peaks = data['peaks'][0]

        data_dict[record] = {'signal': signal,
                             'category': category,
                             'peaks': peaks}

    return data_dict


def load_beat(path, data_dir, half_beat_len):

    information = np.load(path)
    record_id = str(information['record'])
    peak = information['peak']
    cls = information['cls']

    Sigmoid_L = 1.0 / (1 + np.exp(
        -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
            1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

    Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

    raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len: peak + half_beat_len + 1]

    beat = np.multiply(raw_segment, Sigmoid_Win)
    max_v = np.max(beat)
    min_v = np.min(beat)
    beat = beat / (max_v - min_v)

    return beat, cls


def augmentation_transform(path, data_dir, half_beat_len):

    modes = np.array([0, 1, 2])
    mode = np.random.choice(modes, size=1)

    information = np.load(path)
    record_id = str(information['record'])
    peak = information['peak']
    cls = information['cls']

    if mode == 0:

        bias = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign = np.random.choice(np.array([-1, 1]), size=1)

        half_beat_len_ = int(half_beat_len * (1 + sign * bias / 100.0))

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len_ + 1)) - 0.6 * half_beat_len_) * 0.1))

        Sigmoid_R = np.ones((half_beat_len_,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len_)) - 0.6 * half_beat_len_) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len_: peak + half_beat_len_ + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}".format(peak, half_beat_len_)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)

        return beat, cls

    elif mode == 1:

        bias_l = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign_l = np.random.choice(np.array([-1, 1]), size=1)

        bias_r = np.random.randint(low=5, high=10, size=1).astype(np.float32)
        sign_r = np.random.choice(np.array([-1, 1]), size=1)

        half_beat_len_l = int(half_beat_len * (1 + sign_l * bias_l / 100.0))
        half_beat_len_r = int(half_beat_len * (1 + sign_r * bias_r / 100.0))

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len_l + 1)) - 0.6 * half_beat_len_l) * 0.1))

        Sigmoid_R = np.ones((half_beat_len_r,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len_r)) - 0.6 * half_beat_len_r) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        raw_segment = data_dir[record_id + '.mat']['signal'][0, peak - half_beat_len_l: peak + half_beat_len_r + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}, {}".format(peak,
                                                                         half_beat_len_l,
                                                                         half_beat_len_r)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)
        return beat, cls

    elif mode == 2:

        Sigmoid_L = 1.0 / (1 + np.exp(
            -(np.array(np.arange(0, half_beat_len + 1)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_R = np.ones((half_beat_len,)) - 1.0 / (
                1 + np.exp(-(np.array(np.arange(0, half_beat_len)) - 0.6 * half_beat_len) * 0.1))

        Sigmoid_Win = np.concatenate([Sigmoid_L, Sigmoid_R])

        wt_bases = pywt.wavelist(kind='discrete')
        wt_base_name = np.random.choice(wt_bases, size=1)
        # wt_base = pywt.Wavelet('coif2')
        wt_base = pywt.Wavelet(wt_base_name[0])
        # print(wt_base_name)

        record_signal = data_dir[record_id + '.mat']['signal'][0]
        coeffs = pywt.wavedec(record_signal, wt_base, level=11)
        disturb_level = [0, 1, 2, 3]
        for x in disturb_level:
            coeffs[x] = coeffs[x] + 0.5 * np.random.randn(len(coeffs[x]))
        disturbed_record_signal = pywt.waverec(coeffs, wt_base)

        raw_segment = disturbed_record_signal[peak - half_beat_len: peak + half_beat_len + 1]

        assert len(raw_segment) == len(Sigmoid_Win), "{}, {}".format(peak, half_beat_len)

        beat = np.multiply(raw_segment, Sigmoid_Win)
        max_v = np.max(beat)
        min_v = np.min(beat)
        beat = beat / (max_v - min_v)

        return beat, cls


class ECG_TRAIN_DATASET(Dataset):

    def __init__(self, dataset_name, data_loader, data_dict, dataset_split='entire',
                 transform=None, target_transform=None):
        super(ECG_TRAIN_DATASET, self).__init__()

        '''
        The ECG dataset
        dataset_name: The dataset used for training |mitdb|svdb|incartdb|
        data_loader: The function that loads a sample
        data_dict: All records loaded into memory (dict)
        dataset_split: which subset to use for training |entrie(default)|DS1|DS2|
        transform: Transformation function for a sample
        target_transform: Transformation function for labels
        '''

        root_index = '/home/workspace/mingchen/ECG_UDA/data_index'
        categories = ['N', 'V', 'S', 'F']

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.loader = data_loader
        self.data = data_dict
        self.transform = transform
        self.target_transform = target_transform

        self.load_path = osp.join(osp.join(root_index, dataset_name), dataset_split)
        self.data_path = DENOISE_DATA_DIRS[dataset_name]

        self.samples = []

        for cate in categories:
            files = os.listdir(osp.join(self.load_path, cate))
            samples = [(file, cate) for file in files]
            self.samples.extend(samples)

        self.fs = SAMPLE_RATES[dataset_name]
        self.half_beat_len = int(self.fs * 0.7)
        self.fixed_len = 400

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        filename, cate = self.samples[index]
        file_path = osp.join(osp.join(self.load_path, cate), filename)
        # beat, cls = self.loader(file_path, self.data, self.half_beat_len)

        if self.transform is not None:
            beat, cls = self.transform(file_path, self.data, self.half_beat_len)

        else:
            beat, cls = self.loader(file_path, self.data, self.half_beat_len)

        beat = ss.resample(beat, self.fixed_len).astype(np.float32)

        return beat, cls


if __name__ == '__main__':

    data_dict = load_dataset_to_memory(dataset_name='mitdb')

    dataset = ECG_TRAIN_DATASET(dataset_name='mitdb',
                                data_loader=load_beat,
                                data_dict=data_dict,
                                dataset_split='DS1',
                                transform=augmentation_transform)

    beat, cls = dataset.__getitem__(0)

    # plt.figure(0)
    # plt.plot(beat)
    # plt.savefig('../../figures/sample.png')
    # plt.close()

    print(beat.shape)
    print(cls)
