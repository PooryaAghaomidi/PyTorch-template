# -*- coding: utf-8 -*-
"""Data Loader"""

import torch
import numpy as np
from torch.nn import functional as F


class DataGenerator:
    def __init__(self, address, shape, batch_size, cls_num, device):
        self.data = np.load(address)
        self.shape = shape
        self.batch_size = batch_size
        self.cls_num = cls_num
        self.device = device

    def data_generation(self, idx):
        start = idx * self.batch_size

        x_init = self.data[start:start + self.batch_size, 1:]
        y_init = self.data[start:start + self.batch_size, 0]

        x = np.empty((self.batch_size, int(self.shape[2]), int(self.shape[0]), int(self.shape[1])))

        for i, signal in enumerate(x_init):
            x[i, 0, :, :] = signal.reshape((int(self.shape[0]), int(self.shape[1])))

        y = F.one_hot(torch.tensor(y_init, dtype=torch.int64, device=self.device), num_classes=self.cls_num).float()

        return torch.tensor(x, dtype=torch.float32, device=self.device), y
