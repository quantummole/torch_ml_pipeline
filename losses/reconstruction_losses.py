# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class FFTLoss(nn.Module):
    """
    Computes the FFT loss for reconstruction tasks.
    The loss is defined as the mean squared error in the frequency domain.
    The frequency loss is useful to ensure the reconstruction of details(high frequency components) in the input data.
    """
    def __init__(self, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, predictions, targets):
        # Compute FFT of predictions and targets
        pred_fft = torch.fft.fftn(predictions)
        target_fft = torch.fft.fftn(targets)

        # Compute the squared difference in the frequency domain
        loss = torch.pow(torch.abs(pred_fft - target_fft),2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', or 'none'.")

class PerceptualLoss(nn.Module):
    """
    Computes the perceptual loss for reconstruction tasks.
    The loss is defined as the mean squared error between the feature maps of the predictions and targets.
    This loss is useful to ensure that the reconstructed images are perceptually similar to the original images.
    """
    def __init__(self, feature_extractor, reduction='mean'):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        if reduction not in ['mean', 'sum', 'none', 'sum_mean']:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum','sum_mean' or 'none'.")
        self.reduction = reduction

    def forward(self, predictions, targets):
        # Extract features from predictions and targets
        self.feature_extractor.activations.clear()
        _ = self.feature_extractor(predictions)
        pred_features = self.feature_extractor.activations

        self.feature_extractor.activations.clear()
        _ = self.feature_extractor(targets)
        target_features = self.feature_extractor.activations


        # Compute the mean squared error between the feature maps
        losses = [torch.pow(pred_features[i] - target_features[i], 2) for i in range(len(pred_features))]

        if self.reduction == 'mean':
            return torch.mean([loss.mean() for loss in losses])
        elif self.reduction == 'sum':
            return torch.sum([loss.sum() for loss in losses])
        elif self.reduction == 'sum_mean':
            return torch.sum([loss.mean() for loss in losses])
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("Invalid reduction method. Use 'mean', 'sum', 'sum_mean' or 'none'.")


if __name__ == "__main__":
    # Example usage
    predictions = torch.randn(2, 3, 64, 64)  # Batch of 2 images with 3 channels and size 64x64
    targets = torch.randn(2, 3, 64, 64)

    fft_loss_fn = FFTLoss(reduction='mean')
    #perceptual_loss_fn = PerceptualLoss(feature_extractor=None, reduction='mean')  # Replace None with a valid feature extractor

    fft_loss = fft_loss_fn(predictions, targets)
    #perceptual_loss = perceptual_loss_fn(predictions, targets)

    print("FFT Loss:", fft_loss.item())
    #print("Perceptual Loss:", perceptual_loss.item())