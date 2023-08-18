"""Code is adapted from https://github.com/tychovdo/MovingMNIST."""

import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import torch.nn.functional as F
import codecs
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader
from typing import Optional


class HimawariDataset(data.Dataset):
    folder = 'processed'
    # raw_folder = 'raw'
    # processed_folder = 'processed'
    # training_file = 'moving_mnist_train.pt'
    # test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None,
                 post_transform=None, post_target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        # self.transform = transform
        # self.target_transform = target_transform
        # self.post_transform = post_transform
        # self.post_target_transform = post_target_transform
        self.split = split
        self.train = train  # training set or test set

        self.train_data = torch.tensor(np.load(
            os.path.join(self.root, 'himawari_jiaxing_5km_2020.npy')))

        if self.train:
            self.train_data = torch.tensor(np.load(
                os.path.join(self.root, 'himawari_jiaxing_5km_2020.npy')))
        else:
            self.test_data = torch.tensor(np.load(
                os.path.join(self.root, 'himawari_jiaxing_5km_2023.npy')))
        a = 1

    def __getitem__(self, index):

        if self.train:
            seq, target = self.train_data[index: index+10], self.train_data[index+10: index+20]
        else:
            seq, target = self.test_data[index: index+10], self.test_data[index+10: index+20]

        seq = seq.float()
        target = target.float()
        return seq, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        # fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

#
# class NearestInterpTransform:
#     def __init__(self, target_thw, layout='THWC'):
#         """
#
#         Parameters
#         ----------
#         target_thw
#             The target shape with (T, H, W)
#         """
#         self.target_thw = target_thw
#         self.layout = layout
#
#     def __call__(self, data):
#         """
#
#         Parameters
#         ----------
#         data
#             Shape (T, H, W) or (T, H, W, C)
#
#         Returns
#         -------
#         rescaled_data
#             Shape (T, H, W, C)
#             C will be 1 if the input data shape is (T, H, W)
#         """
#         if self.target_thw == data.shape:
#             return data.view(*tuple(self.target_thw + (1,)))
#         else:
#             assert len(data.shape) == 3
#             rescaled_data = F.interpolate(data.view((1, 1) + data.shape), self.target_thw, mode='nearest')
#             rescaled_data = rescaled_data.view(self.target_thw + (1,))
#             print('rescaled_data.shape=', rescaled_data.shape)
#             return rescaled_data


class HimawariDataModule(pl.LightningDataModule):
    def __init__(self,
                 root: str = None,
                 val_ratio=0.33, seed=123, batch_size: int = 32,
                 rescale_input_shape=None, rescale_target_shape=None):
        """

        Parameters
        ----------
        root
        val_ratio
        batch_size
        rescale_input_shape
            For the purpose of testing. Rescale the inputs
        rescale_target_shape
            For the purpose of testing. Rescale the targets
        """
        super().__init__()
        if root is None:
            from ...config import cfg
            root = os.path.join(cfg.datasets_dir, "himawari")
        self.root = root
        self.val_ratio = val_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.rescale_input_shape = rescale_input_shape
        self.rescale_target_shape = rescale_target_shape

        # if self.rescale_input_shape is None:
        #     self.post_transform = NearestInterpTransform(target_thw=(10, 64, 64))
        # else:
        #     self.post_transform = NearestInterpTransform(target_thw=self.rescale_input_shape)
        #
        # if self.rescale_target_shape is None:
        #     self.post_target_transform = NearestInterpTransform(target_thw=(10, 64, 64))
        # else:
        #     self.post_target_transform = NearestInterpTransform(target_thw=self.rescale_target_shape)

    def prepare_data(self):
        HimawariDataset(self.root, train=True, download=True)
        HimawariDataset(self.root, train=False, download=True)

    @property
    def input_shape(self):
        """

        Returns
        -------
        ret
            Contains (T, H, W, C)
        """
        if self.rescale_input_shape is not None:
            return self.rescale_input_shape + (1,)
        else:
            return 10, 64, 64, 4

    @property
    def target_shape(self):
        """

        Returns
        -------

        """
        if self.rescale_target_shape is not None:
            return self.rescale_target_shape + (1,)
        else:
            return 10, 64, 64, 4

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_val_data = HimawariDataset(self.root, train=True)
            all_indices = range(len(train_val_data))
            train_indices, val_indices = train_test_split(all_indices, test_size=self.val_ratio, random_state=self.seed)
            self.moving_mnist_train = Subset(train_val_data, train_indices)
            self.moving_mnist_val = Subset(train_val_data, val_indices)

        if stage == "test" or stage is None:
            self.moving_mnist_test = HimawariDataset(self.root, train=False)

        if stage == "predict" or stage is None:
            self.moving_mnist_predict = HimawariDataset(self.root, train=False)

    def train_dataloader(self):
        return DataLoader(self.moving_mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.moving_mnist_val, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.moving_mnist_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.moving_mnist_predict, batch_size=self.batch_size, shuffle=False, num_workers=4)

    @property
    def num_train_samples(self):
        return len(self.moving_mnist_train)

    @property
    def num_val_samples(self):
        return len(self.moving_mnist_val)

    @property
    def num_test_samples(self):
        return len(self.moving_mnist_test)

    @property
    def num_predict_samples(self):
        return len(self.moving_mnist_predict)
