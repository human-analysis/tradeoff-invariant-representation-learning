# gaussian.py

import math
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np

__all__ = ['Gaussian4d']


class Gaussian(pl.LightningDataModule):

    def __init__(self, root, noise, train, val):
        # dim = 10
        # n_train = 6000
        # sigma = 1
        # mean = 0.0 * torch.ones(dim, )
        # torch.manual_seed(100)
        # ps = torch.distributions.MultivariateNormal(mean, sigma * torch.eye(dim))
        # data_s = ps.sample((n_train, 1)).squeeze(1)

        def thresh2(y, th):
            y_ = torch.zeros_like(y)
            th_mask = y > th
            y_[th_mask] = 1
            y_int = y_[:, 3] * 8 + y_[:, 2] * 4 + y_[:, 1] * 2 + y_[:, 0]
            y_onehot = torch.zeros(y.shape[0], 2 ** y.shape[1], device=y.device).scatter_(1, y_int.unsqueeze(
                1).long(), 1)

            return y_onehot, y_int

        if train:
            data_train = np.load(root + '/data_train_20.npy', allow_pickle=True)
            data = torch.from_numpy(data_train).float()

            # self.y = torch.cos(np.pi / 6 * data[:, 0:2])
            # th1 = 0.986
            # th2 = 0.938
            # th3 = 0.82
            # self.y_onehot, self.y_int = thresh(self.y, th1, th2, th3)

            self.y = torch.cos(np.pi / 6 * data[:, 0:4])
            th = 0.9383
            self.y_onehot, self.y_int = thresh2(self.y, th)
            self.x = torch.cos(np.pi / 6 * data[:, 0:4]) + noise * data[:, 10:14]
            self.s1 = torch.sin(np.pi / 6 * data[:, 0:2])
            self.s2 = torch.cos(np.pi / 6 * data[:, 2:4])
            self.s = torch.cat((self.s1, self.s2), dim=1)
            self.label = data[:, 20]

        elif val:
            data_val = np.load(root + '/data_valid_20.npy', allow_pickle=True)
            data = torch.from_numpy(data_val).float()

            # self.y = torch.cos(np.pi / 6 * data[:, 0:2])
            # th1 = 0.986
            # th2 = 0.938
            # th3 = 0.82
            # self.y_onehot, self.y_int = thresh(self.y, th1, th2, th3)

            self.y = torch.cos(np.pi / 6 * data[:, 0:4])
            th = 0.9383
            self.y_onehot, self.y_int = thresh2(self.y, th)
            self.x = torch.cos(np.pi / 6 * data[:, 0:4]) + noise * data[:, 10:14]
            self.s1 = torch.sin(np.pi / 6 * data[:, 0:2])
            self.s2 = torch.cos(np.pi / 6 * data[:, 2:4])
            self.s = torch.cat((self.s1, self.s2), dim=1)
            self.label = data[:, 20]
        else:
            data_test = np.load(root + '/data_test_20.npy', allow_pickle=True)
            data = torch.from_numpy(data_test).float()

            # self.y = torch.cos(np.pi / 6 * data[:, 0:2])
            # th1 = 0.986
            # th2 = 0.938
            # th3 = 0.82
            # self.y_onehot, self.y_int = thresh(self.y, th1, th2, th3)

            self.y = torch.cos(np.pi / 6 * data[:, 0:4])
            th = 0.9383
            self.y_onehot, self.y_int = thresh2(self.y, th)
            self.x = torch.cos(np.pi / 6 * data[:, 0:4]) + noise * data[:, 10:14]
            self.s1 = torch.sin(np.pi / 6 * data[:, 0:2])
            self.s2 = torch.cos(np.pi / 6 * data[:, 2:4])
            self.s = torch.cat((self.s1, self.s2), dim=1)
            self.label = data[:, 20]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y, s, label = self.x[index], self.y_onehot[index], self.s[index], self.y_int[index]
        return x, y, s, label


class Gaussian4d(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True


        def thresh2(y, th):
            y_ = torch.zeros_like(y)
            th_mask = y > th
            y_[th_mask] = 1
            y_int = y_[:, 3] * 8 + y_[:, 2] * 4 + y_[:, 1] * 2 + y_[:, 0]
            y_onehot = torch.zeros(y.shape[0], 2 ** y.shape[1], device=y.device).scatter_(1, y_int.unsqueeze(
                1).long(), 1)
            return y_onehot, y_int

        data_train = np.load(self.opts.feature_path + '/data_train_20.npy', allow_pickle=True)
        data = torch.from_numpy(data_train).float()

        # self.y = torch.cos(np.pi / 6 * data[:, 0:2])
        # th1 = 0.986
        # th2 = 0.938
        # th3 = 0.82
        # self.y_onehot, self.y_int = thresh(self.y, th1, th2, th3)

        self.y = torch.cos(np.pi / 6 * data[:, 0:4])
        th = 0.9383
        self.y_onehot, self.y_int = thresh2(self.y, th)
        self.x = torch.cos(np.pi / 6 * data[:, 0:4]) + self.opts.noise * data[:, 10:14]
        self.s1 = torch.sin(np.pi / 6 * data[:, 0:2])
        self.s2 = torch.cos(np.pi / 6 * data[:, 2:4])
        self.s = torch.cat((self.s1, self.s2), dim=1)
        self.label = data[:, 20]


    def train_dataloader(self):
        batch_size = self.opts.batch_size_train
        if self.opts.dataset_type == 'Gaussian':
            dataset = Gaussian(root=self.opts.feature_path, noise=self.opts.noise, train=True, val=False)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        batch_size = self.opts.batch_size_test
        if self.opts.dataset_type == 'Gaussian':
            dataset = Gaussian(root=self.opts.feature_path, noise=self.opts.noise, train=False, val=True)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        batch_size = self.opts.batch_size_test
        if self.opts.dataset_type == 'Gaussian':
            dataset = Gaussian(root=self.opts.feature_path, noise=self.opts.noise, train=False, val=False)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_train_dataloader(self):
        batch_size = self.opts.batch_size_test
        if self.opts.dataset_type == 'Gaussian':
            dataset = Gaussian(root=self.opts.feature_path,  noise=self.opts.noise, train=True, val=False)
        else:
            print('Gaussian dataset type does not exist')
            dataset = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader