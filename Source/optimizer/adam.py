# -*- coding: utf-8 -*-
"""Adam optimizer"""

import torch.optim as optim


def adam_opt(model, lr, weight_decay=0):
    return optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay,
                      amsgrad=False, foreach=None, maximize=False, capturable=False, differentiable=False)
