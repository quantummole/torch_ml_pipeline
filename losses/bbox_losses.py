# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class CIoULoss(nn.Module):
    """
    The center loss gives a gradient even when the boxees are not overlapping.
    The aspect ratios are computed using both atan and acot functions to account for narrow boxes that may have a very small width or height.
    """
    def __init__(self, reduction='mean'):
        super(CIoULoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be either 'mean' or 'sum'")
        self.reduction = reduction
    def forward(self, pred, target):
        assert pred.shape == target.shape, "pred and target must have the same shape"
        pred_cent = (pred[..., :2] + pred[..., 2:])/2
        target_cent = (target[..., :2] + target[..., 2:])/2
        pred_wh = pred[..., 2:] - pred[..., :2]
        target_wh = target[..., 2:] - target[..., :2]

        
        pred_atanwh = torch.atan2(pred_wh[..., 1], pred_wh[..., 0])
        target_atanwh = torch.atan2(target_wh[..., 1], target_wh[..., 0])
        atanwh_loss = torch.pow(pred_atanwh -  target_atanwh,2)*4/torch.pi**2

        pred_acotwh = torch.atan2(pred_wh[..., 0], pred_wh[..., 1])
        target_acotwh = torch.atan2(target_wh[..., 0], target_wh[..., 1])
        acotwh_loss = torch.pow(pred_acotwh-target_acotwh,2)*4/torch.pi**2

        intersection = torch.min(pred[..., 2:], target[..., 2:]) - torch.max(pred[..., :2], target[..., :2])
        intersection = torch.clamp(intersection, min=1e-6)
        union = torch.max(pred[..., 2:], target[..., 2:]) - torch.min(pred[..., :2], target[..., :2])
        union = torch.clamp(union, min=1e-6)
        iou_loss = 1 - intersection[..., 0] * intersection[..., 1] / (union[..., 0] * union[..., 1])

        convex_hull_diagonal = union[..., 0]**2 +  union[..., 1]**2
        center_loss = torch.pow(pred_cent - target_cent,2).sum(dim=-1)/convex_hull_diagonal

        loss = iou_loss + center_loss + atanwh_loss + acotwh_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError("reduction must be either 'mean', 'sum' or 'none'")
        return loss
    
if __name__ == "__main__":
    pred = torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32)
    target = torch.tensor([[0, 0, 2, 2], [4, 4, 6, 6]], dtype=torch.float32)
    loss_fn = CIoULoss(reduction='none')
    loss = loss_fn(pred, target)
    print("CIoU Loss:", loss)
            

