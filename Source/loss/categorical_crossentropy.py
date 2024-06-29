# -*- coding: utf-8 -*-
"""Categorical loss function"""

from torch import nn


def cross_entropy(label_smoothing=0.0):
    return nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean',
                               label_smoothing=label_smoothing)
