# celeba_feature_loader.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import torch
from random import shuffle

__all__ = ['FeatureLoaderCelebAHQ']


class FeatureCelebAHQ:
    def __init__(self, filespath=None, partition='train'):

        if partition == 'train':
            print('Loading train data ...')
            self.z = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'z_train.out')))
            self.y = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'y_train.out')))
            self.s = torch.from_numpy(np.loadtxt(os.path.join(filespath, 's_train.out')))
            print('Loading is done!')
        elif partition == 'val':
            self.z = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'z_val.out')))
            self.y = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'y_val.out')))
            self.s = torch.from_numpy(np.loadtxt(os.path.join(filespath, 's_val.out')))
        elif partition == 'test':
            self.z = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'z_test.out')))
            self.y = torch.from_numpy(np.loadtxt(os.path.join(filespath, 'y_test.out')))
            self.s = torch.from_numpy(np.loadtxt(os.path.join(filespath, 's_test.out')))

    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        # print(f"z={self.z.shape}, y={self.y.shape}, s={self.s.shape}")
        z = self.z[index].float()
        y = self.y[index].long()
        s = self.s[index]

        return z, y, s


class FeatureLoaderCelebAHQ(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = FeatureCelebAHQ(
            filespath=self.opts.features_path,
            partition='train'
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        dataset = FeatureCelebAHQ(
            filespath=self.opts.features_path,
            partition='val'
        )

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        dataset = FeatureCelebAHQ(
            filespath=self.opts.features_path,
            partition='test'
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader
