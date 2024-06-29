# -*- coding: utf-8 -*-
"""Device function"""

import torch
import warnings


def set_gpu():
    print('The version of tensorflow is: \n', torch.__version__)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        warnings.warn('GPU is not available')

    print('Device is: ', device)

    return device
