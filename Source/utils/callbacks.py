# -*- coding: utf-8 -*-
"""Callbacks"""

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def callback():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    return SummaryWriter('checkpoints/tensorboard_{}'.format(timestamp)), 'checkpoints/model_{}'.format(timestamp + '.pt')
