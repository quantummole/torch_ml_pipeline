# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class SimpleGaussianLoss(nn.Module):
    """
    Computes the Gaussian loss for regression tasks.
    The loss is defined as the negative log likelihood of a Gaussian distribution.
    In case the predictions are of form NxD assumes that the D terms are independent.
    """
    def __init__(self, reduction='mean'):
        super(SimpleGaussianLoss, self).__init__()
        self.reduction = reduction

    def forward(self, predictions, targets):
        means = predictions[:, 0]
        sds = predictions[:, 1].clamp(min=1e-7) 
        loss = 0.5*(torch.log(2*torch.pi*sds*sds) + ((targets - means)**2)/sds**2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss