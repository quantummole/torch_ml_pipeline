# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn

class SeparableConv(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size,  dilation=1, conv_type=nn.Conv2d, padding=None) :
        super(SeparableConv, self).__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
            conv_type(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels),
            conv_type(in_channels, out_channels, kernel_size=1),
        )   
    def forward(self, x):
        return self.conv(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  dilation=1, conv_type=nn.Conv2d, padding=None) :
        super(Conv, self).__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = conv_type(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    def forward(self, x):
        return self.conv(x)
    
class DoubleConv(nn.Module) :
    """
    Implements a double convolution block that applies two consecutive convolutional layers with an activation function in between.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.SiLU, conv_type=nn.Conv2d) :
        super(DoubleConv, self).__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Sequential(
            conv_type(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            activation(inplace=True),
            conv_type(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            activation(inplace=True)  
        )
    def forward(self, x):
        return self.conv(x)
    
class ResidualBlock(nn.Module) :
    """
    Residual Block that applies a DoubleConv layer and adds the input to the output.
    The shortcut connection allows for better gradient flow and helps to mitigate the vanishing gradient problem.
    Was used in ResNet architectures to enable training of very deep networks.
    Residual Connections effectively modify the network into an ensemble of many networks of various depths.
    The residual computation allows the network to learn an identity function if needed which can help us avoid tuning the depth parameter.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.SiLU, conv_type=nn.Conv2d) :
        super(ResidualBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dilation, activation, conv_type)
        self.shortcut = conv_type(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()      
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    
class DenseBlock(nn.Module) :
    """
    Dense Block that concatenates the input with the output of a DoubleConv layer.
    This block has the benefits of preserving lower level features while allowing for deeper feature extraction.
    The feature reuse also helps with gradient flow, making it easier to train deeper networks.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=nn.SiLU, conv_type=nn.Conv2d) :
        super(DenseBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, dilation, activation, conv_type)     
    def forward(self, x):
        return torch.cat([x, self.conv(x)], dim=1)
    
class InceptionBlock(nn.Module) :
    """
    The basic version of Inception Block that applies multiple convolutional branches with different kernel sizes.
    Each branch processes the input independently, and their outputs are concatenated.
    This mimic the image processing principle of using multiple receptive fields to capture features at different scales.
    The output is then passed through a 1x1 convolution to combine the features from all branches.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, dilation=1, activation=nn.SiLU, conv_type=nn.Conv2d) :
        super(InceptionBlock, self).__init__()
        reduction_factor = len(kernel_sizes)
        self.branches = nn.ModuleList([
            DoubleConv(in_channels, out_channels // reduction_factor, k, dilation=dilation, activation=activation, conv_type=conv_type)
            for k in kernel_sizes
        ])   
        input_size = (out_channels // reduction_factor)*reduction_factor
        self.combiner = conv_type(input_size, out_channels, kernel_size=1)
    def forward(self, x):
        return self.combiner(torch.cat([branch(x) for branch in self.branches], dim=1))

class C3Block(nn.Module):
    """
    This is similar to the InceptionBlock but uses a different approach to do multiscale feature extraction.
    It applies kernels with different dilations to capture features at different scales mimicking a feature pyramid    
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilations=1, activation=nn.SiLU, conv_type=nn.Conv2d) :
        super(C3Block, self).__init__()
        reduction_factor = len(dilations) + 1
        self.concentration = SeparableConv(in_channels, out_channels//reduction_factor, kernel_size, dilation=1, activation=activation, conv_type=conv_type)
        self.branches = nn.ModuleList([
            SeparableConv(out_channels//reduction_factor, out_channels//reduction_factor, kernel_size, dilation=dilation, activation=activation, conv_type=conv_type)
            for dilation in dilations
        ])   
        input_size = (out_channels//reduction_factor)*reduction_factor
        self.combiner = conv_type(input_size, out_channels, kernel_size=1)
    def forward(self, x):
        intermediate = self.concentration(x)
        return self.combiner(torch.cat([branch(intermediate) for branch in self.branches]+[intermediate], dim=1))


class SqueezeExciteBlock(nn.Module):
    """
    Computes Squeeze-and-Excitation (SE) block for feature recalibration.
    It is a channel attention mechanism that adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_channels, reduction=16, activation=nn.SiLU):
        super(SqueezeExciteBlock, self).__init__()
        self.scale = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            activation(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid
        )
    def forward(self, x):
        y = x.flatten(start_dim=2).mean(dim=2)
        scale = self.scale(y).view(*x.size())
        x = x * scale
        return x
    
class SAMBlock(nn.Module):
    """
    Computes Spatial Attention Module (SAM) for feature recalibration.
    """
    def __init__(self, in_channels, reduction=16, activation=nn.SiLU, conv_type=nn.Conv2d):
        super(SAMBlock, self).__init__()
        self.scale = nn.Sequential(
            conv_type(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_x = x.mean(dim=1,keepdim=True)
        max_x = x.max(dim=1,keepdim=True)[0]
        inp = torch.cat([avg_x, max_x], dim=1)
        x = x * self.scale(inp)
        return x
    
if __name__ == "__main__":
    import numpy as np
    import time
    
    IN_CHANNELS = 3
    OUT_CHANNELS = 16
    SIZE = 32
    
    for conv_type in [nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d]:
        if conv_type == nn.Conv3d:
            test_input = torch.randn(1, IN_CHANNELS, SIZE, SIZE, SIZE)
        else:
            test_input = torch.randn(1, IN_CHANNELS, SIZE, SIZE)
        for block_class in [DoubleConv, ResidualBlock, DenseBlock, InceptionBlock, SeparableConv, Conv]:
            print(f"Testing with {conv_type.__name__} and {block_class.__name__}...")

            if block_class == InceptionBlock:
                block = block_class(IN_CHANNELS, OUT_CHANNELS, kernel_sizes=[3, 5, 7], conv_type=conv_type)
            else:
                block = block_class(IN_CHANNELS, OUT_CHANNELS, kernel_size=3, conv_type=conv_type)

            start_time = time.time()
            output = block(test_input)
            end_time = time.time()
            sizes = output.size()

            if block_class == DenseBlock:
                assert sizes[1] == OUT_CHANNELS + IN_CHANNELS and max(sizes[2:]) == SIZE and min(sizes[2:]) == SIZE , (test_input.size(),sizes)
            else:
                assert sizes[1] == OUT_CHANNELS and max(sizes[2:]) == SIZE and min(sizes[2:]) == SIZE , (test_input.size(),sizes)
            print(f"{block_class.__name__} Time: {end_time - start_time:.6f} seconds")
            