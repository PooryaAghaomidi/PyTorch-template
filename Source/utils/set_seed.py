# -*- coding: utf-8 -*-
"""Seed function"""

import torch
import random
import numpy as np


def set_seed(device, seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed_value)

    return generator
