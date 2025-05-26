# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc


class FocalCELoss(nn.Module) :
    """
    Key idea is to throttle the weights for the correct high confident predictions to zero.
    The throttling is done by multiplying the weights with (1 - predictions)^gamma.
    There are also weight factors computed for class rebalancing
    """
    def __init__(self,gamma = 0, temperature = 1.) :
        super(FocalCELoss,self).__init__()
        self.gamma = gamma
        self.temperature = temperature
    def forward(self,logits,target, reduce="sum") :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)/self.temperature
        target = target.view(logits.shape[0],-1)
        outputs_onehot = torch.zeros_like(logits)
        outputs_onehot = outputs_onehot.scatter(1,target.view(logits.shape[0],1,-1),1.0)
        predictions = tfunc.softmax(logits,dim=1)
        class_counts = outputs_onehot.sum(dim=0,keepdim=True).sum(dim=-1,keepdim=True)
        weights = outputs_onehot.sum()/torch.clamp(class_counts,1e-7)
        factor = (1- predictions).pow(self.gamma)
        loss = tfunc.nll_loss(tfunc.log_softmax(logits,dim=1)*factor*weights,target, reduction=reduce)
        return loss

class FocalBCELoss(nn.Module) :
    """
    Key idea is to throttle the weights for the correct high confident predictions to zero.
    The throttling is done by multiplying the weights with (1 - predictions)^gamma.
    There are also weight factors computed for class rebalancing
    """
    def __init__(self,gamma = 0, temperature = 1.) :
        super(FocalCELoss,self).__init__()
        self.gamma = gamma
        self.temperature = temperature
    def forward(self,logits,target, reduce="sum") :
        logits = logits.view(logits.shape[0],-1)/self.temperature
        target = target.view(logits.shape[0],-1)
        
        pos_count = target.sum(dim=0,keepdim=True)
        neg_count = (1-target).sum(dim=0,keepdim=True)
        class_counts = pos_count + neg_count
        weights = target*class_counts/pos_count + (1-target)*class_counts/neg_count

        predictions = torch.sigmoid(logits)
        predictions = predictions*target + (1-predictions)*(1-target)
        predictions = predictions.clamp(min=1e-7, max=1-1e-7)
        factor = (1- predictions).pow(self.gamma)

        focal_likelihood = weights*factor*torch.log(prediction)
        if reduce == "mean" :
            loss = -1*focal_likelihood.mean()
        elif reduce == "sum" :
            loss = -1*focal_likelihood.sum()
        elif reduce == "none" :
            loss = -1*focal_likelihood               
        else :
            raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")
        return loss


