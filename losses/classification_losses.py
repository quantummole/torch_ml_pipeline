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
    def __init__(self,gamma = 0, temperature = 1., reduction="sum") :
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")
        super(FocalCELoss,self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.reduction = reduction
    def forward(self,logits,target) :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)/self.temperature
        target = target.view(logits.shape[0],-1)
        outputs_onehot = torch.zeros_like(logits)
        outputs_onehot = outputs_onehot.scatter(1,target.view(logits.shape[0],1,-1),1.0)
        predictions = tfunc.softmax(logits,dim=1)
        class_counts = outputs_onehot.sum(dim=0,keepdim=True).sum(dim=-1,keepdim=True)
        weights = outputs_onehot.sum()/torch.clamp(class_counts,1e-7)
        factor = (1- predictions).pow(self.gamma)
        loss = tfunc.nll_loss(tfunc.log_softmax(logits,dim=1)*factor*weights,target, reduction=self.reduction)
        return loss

class FocalBCELoss(nn.Module) :
    """
    Key idea is to throttle the weights for the correct high confident predictions to zero.
    The throttling is done by multiplying the weights with (1 - predictions)^gamma.
    There are also weight factors computed for class rebalancing
    """
    def __init__(self,gamma = 0, temperature = 1., reduction="sum") :
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")
        super(FocalBCELoss,self).__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.reduction = reduction
    def forward(self,logits,target) :
        logits = logits.view(logits.shape[0],-1)/self.temperature
        target = target.view(logits.shape[0],-1)
        assert target.shape[1] == logits.shape[1], "Target and logits must have the same dimensions."
        
        pos_count = target.sum(dim=0,keepdim=True)
        neg_count = (1-target).sum(dim=0,keepdim=True)
        class_counts = pos_count + neg_count
        weights = target*class_counts/pos_count + (1-target)*class_counts/neg_count

        predictions = torch.sigmoid(logits)
        predictions = predictions*target + (1-predictions)*(1-target)
        predictions = predictions.clamp(min=1e-7, max=1-1e-7)
        factor = (1- predictions).pow(self.gamma)

        focal_likelihood = weights*factor*torch.log(predictions)
        if self.reduction == "mean" :
            loss = -1*focal_likelihood.mean()
        elif self.reduction == "sum" :
            loss = -1*focal_likelihood.sum()
        elif self.reduction == "none" :
            loss = -1*focal_likelihood               
        else :
            raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")
        return loss


if __name__ == "__main__":
    import numpy as np
    test_logits = torch.from_numpy(np.array([[0,0],[0,10],[-10,0],[10,-10]]).astype(np.float32))
    target = torch.from_numpy(np.array([0,1,0]).astype(np.int64))
    target_bce = torch.from_numpy(np.array([[1,0],[0,1],[1,0],[0,1]]).astype(np.float32))
    baseline_ce = nn.CrossEntropyLoss(reduction="none")
    baseline_bce = nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_fn = FocalCELoss(gamma=2, reduction="none")
    bce_loss_fn = FocalBCELoss(gamma=2, reduction="none")
    
    baseline_ce_loss = baseline_ce(test_logits, target,)
    baseline_bce_loss = baseline_bce(test_logits, target_bce.float())
    print("Baseline CE Loss:", baseline_ce_loss)
    print("Baseline BCE Loss:", baseline_bce_loss)

    ce_loss = ce_loss_fn(test_logits, target)
    bce_loss = bce_loss_fn(test_logits,target_bce)

    print("CE Loss:", ce_loss)
    print("BCE Loss:", bce_loss)

    ce_loss_fn = FocalCELoss(gamma=0, reduction="none")
    bce_loss_fn = FocalBCELoss(gamma=0, reduction="none")

    ce_loss = ce_loss_fn(test_logits, target)
    bce_loss = bce_loss_fn(test_logits,target_bce)

    print("CE Loss gamma=0:", ce_loss)
    print("BCE Loss gamma=0:", bce_loss)
