# folk_feature_loader.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import numpy as np
import os
import torch
from random import shuffle

from sklearn.model_selection import train_test_split

__all__ = ['FeatureLoaderFolk_embedding_4class']


class FeatureFolk:
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
        self.s = data['s']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        x = self.x[index].float()
        y = self.y[index].long()
        s = self.s[index]

        # 1to9 -> 0to8
        s -= 1

        return x, y, s


class FeatureLoaderFolk_embedding_4class(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

        self.data = self.split_data()

    def split_data(self):
        print('Loading train data ...')
        x = torch.from_numpy(np.loadtxt(os.path.join(self.opts.features_path, 'features_embedded.out'))).float()
        y = torch.from_numpy(np.loadtxt(os.path.join(self.opts.features_path, 'label.out')))
        if self.opts.sensitive_attr == 'race':
            s = torch.from_numpy(np.loadtxt(os.path.join(self.opts.features_path, 'group_race.out')))
        else:
            s = torch.from_numpy(np.loadtxt(os.path.join(self.opts.features_path, 'group.out'))).unsqueeze(1)

        # Combine y classes from 6 classes to 4 classes :: 0, 1, {2,3,4,5}, 6
        y[y == 3] = 2
        y[y == 4] = 2
        y[y == 5] = 2

        y[y == 6] = 3

        x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=0.3,
                                                                             random_state=self.opts.manual_seed)

        n_test = int(0.5 * len(x_test))

        data = {}
        data['train'] = {'x': x_train, 'y': y_train, 's': s_train}
        data['val'] = {'x': x_test[:n_test], 'y': y_test[:n_test], 's': s_test[:n_test]}
        data['test'] = {'x': x_test[n_test:], 'y': y_test[n_test:], 's': s_test[n_test:]}
        print('Loading is done!')

        # import pdb; pdb.set_trace()

        return data

    def train_dataloader(self):
        dataset = FeatureFolk(self.data['train'])

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self):
        dataset = FeatureFolk(self.data['val'])

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self):
        dataset = FeatureFolk(self.data['test'])

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory
        )
        return loader



