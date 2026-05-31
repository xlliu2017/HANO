"""Data loading and preprocessing utilities."""

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy import interpolate


def get_interp2d(x, n_f, n_c):
    x_f, y_f = np.linspace(0, 1, n_f), np.linspace(0, 1, n_f)
    x_c, y_c = np.linspace(0, 1, n_c), np.linspace(0, 1, n_c)
    x_interp = []
    for i in range(len(x)):
        xi_interp = interpolate.interp2d(x_f, y_f, x[i])
        x_interp.append(xi_interp(x_c, y_c))
    return torch.tensor(np.stack(x_interp, axis=0), dtype=torch.float32)


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super().__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except (OSError, ValueError, NotImplementedError):
            self.data = h5py.File(self.file_path, "r")
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


def Data_load(path, nsample, res_input, res_output, xGN=True, xnormalizer=None, train_data=True):
    reader = MatReader(path)

    if train_data:
        x = reader.read_field("coeff")[:nsample, :, :]
        y = reader.read_field("sol")[:nsample, :, :]
    else:
        x = reader.read_field("coeff")[-nsample:, :, :]
        y = reader.read_field("sol")[-nsample:, :, :]
    res_datasets = x.shape[2]
    s1 = int((res_datasets - 1) / (res_input - 1))
    x = x[:, ::s1, ::s1]
    y = get_interp2d(y, res_datasets, res_output)

    if train_data:
        xnormalizer = UnitGaussianNormalizer(x)
        ynormalizer = UnitGaussianNormalizer(y)
    else:
        ynormalizer = None

    if xGN:
        x = xnormalizer.encode(x)

    return torch.unsqueeze(x, dim=1), y, xnormalizer, ynormalizer


def Data_NS(path, nsample, T_in, T, train_data=True):
    reader = MatReader(path)
    if train_data:
        x = reader.read_field("u")[:nsample, :, :, : T + T_in]
        y = reader.read_field("u")[:nsample, :, :, T_in : T + T_in]
    else:
        x = reader.read_field("u")[-nsample:, :, :, :T_in]
        y = reader.read_field("u")[-nsample:, :, :, T_in : T + T_in]

    return x, y


class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super().__init__()

        self.register_buffer("mean", torch.mean(x, 0))
        self.register_buffer("std", torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps
                mean = self.mean[:, sample_idx]

        x = (x * std) + mean
        return x
