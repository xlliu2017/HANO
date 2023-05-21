import torch
import numpy as np
import torch.nn as nn
import scipy.io
import h5py
import pickle
from scipy import interpolate

def get_interp2d(x, n_f, n_c):
    '''
    interpolate (N, n_f, n_f) to (N, n_c, n_c)
    '''
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
        except:
            self.data = h5py.File(self.file_path, 'r')
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
    '''
    Input:
        path: data path, 'data/.mat';
        nsample: number of data, 1000;
        res_input: resolution of the input data, 512;   res_output: resolution of the output data, 526; (res_input > res_output means more information is used)
        xGN: whether normalize x, True or False;
        xnormalizer: when xGN=True, train_data create xnormalizer, apply it on the val_data and the test_data;
        train_data: train_data or not, True or False
    Output:
        x: [bsz, 1, res_input, res_input];
        y: [bsz, res_output, res_output];
        xnormalizer: function;
        ynormalizer: function;
    '''
    reader = MatReader(path)

    if train_data:
        x = reader.read_field('coeff')[:nsample, :, :]
        y = reader.read_field('sol')[:nsample, :, :]
    else:
        x = reader.read_field('coeff')[-nsample:, :, :]
        y = reader.read_field('sol')[-nsample:, :, :]
    res_datasets = x.shape[2]
    s1 = int((res_datasets-1)/(res_input-1))
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
        x = reader.read_field('u')[:nsample, :, :, :T + T_in]
        y = reader.read_field('u')[:nsample, :, :, T_in:T + T_in]
    else:
        x = reader.read_field('u')[-nsample:, :, :, :T_in]
        y = reader.read_field('u')[-nsample:, :, :, T_in:T + T_in]

    return x, y

class UnitGaussianNormalizer(nn.Module):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.register_buffer('mean', torch.mean(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.register_buffer('std', torch.std(x, 0))
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"

def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)



