# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class SoftDiceLoss(nn.Module):
    """
    The Dice Loss is a measure of overlap between two samples.
    It is often used in image segmentation tasks to evaluate the performance of a model.
    The loss is computed as 1 - Dice Coefficient, where the Dice Coefficient is defined as:
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    where |X| and |Y| are the cardinalities of sets X and Y, and |X ∩ Y| is the cardinality of their intersection.
    Alternatively, the loss can be computed as -log(Dice Coefficient). 
    This is useful when the Dice Coefficient is close to 0 since the gradients would be larger in this case, allowing the model to learn faster.
    The loss can be reduced using different methods: "mean", "sum", "none", or "dice".
    """
    def __init__(self, smooth=1e-5, use_log=True, reduction="mean"):
        super(SoftDiceLoss,self).__init__()
        if reduction not in ["mean", "sum", "none", "dice"]:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', 'none', or 'dice'.")
        self.smooth = smooth
        self.use_log = use_log
        self.reduction = reduction
    def forward(self, predictions, target) :
        target = target.view(predictions.shape[0],predictions.shape[1],-1)
        predictions = predictions.view(predictions.shape[0],predictions.shape[1],-1)
        intersection = torch.sum(predictions * target, dim=2)
        union = torch.sum(predictions, dim=2) + torch.sum(target, dim=2)
        dice = 2*(intersection + self.smooth)/(union + self.smooth)
        if self.use_log:
            diceloss = -torch.log(dice)
        else:
            diceloss = 1 - dice
        if self.reduction == "mean" :
            loss = diceloss.mean()
        elif self.reduction == "sum" :
            loss = diceloss.sum()
        elif self.reduction == "none" :
            loss = diceloss     
        elif self.reduction == "dice":
            loss = dice          
        else :
            raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")
        return loss

class BoundaryLoss(nn.Module):
    """
    This loss function is designed to focus on the boundaries of the target segmentation masks.
    It computes the boundary of the target mask using a max pooling operation and then applies the SoftDiceLoss
    """
    def __init__(self, smooth=1e-5, use_log=True, maxpool_kernel_size=3, maxpool_type=nn.MaxPool2d, reduction="mean"):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")
        super(BoundaryLoss,self).__init__()
        self.smooth = smooth
        self.diceloss = SoftDiceLoss(smooth=smooth, use_log=use_log, reduction=reduction)
        self.maxpool = maxpool_type(kernel_size=maxpool_kernel_size, stride=1, padding=maxpool_kernel_size//2)
    def forward(self, predictions, target) :
        target_boundary_internal = target + self.maxpool(-1*target)
        target_boundary_external = self.maxpool(target) - target
        target_boundary = torch.clamp(target_boundary_internal + target_boundary_external, min=0.0, max=1.0)
        predictions_boundary = predictions*target_boundary
        loss = self.diceloss(predictions_boundary, target_boundary_internal)
        return loss


if __name__ == "__main__":
    import numpy as np
    test_predictions = torch.sigmoid(torch.randn(3, 2, 20, 20).float())
    test_target = torch.zeros_like(test_predictions)
    test_target[:, 1, 5:15, 5:15] = 1.0  # Create a square mask in the second channel
    
    dice_loss_fn = SoftDiceLoss(reduction="mean")
    boundary_loss_fn = BoundaryLoss(reduction="mean")

    dice_loss = dice_loss_fn(test_predictions, test_target)
    boundary_loss = boundary_loss_fn(test_predictions, test_target)

    print("Dice Loss:", dice_loss.item())
    print("Boundary Loss:", boundary_loss.item())