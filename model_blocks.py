# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:52:38 2018

@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class DoubleConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_size = 3,dilation = 1) :
        super(DoubleConvLayer,self).__init__()
        padding = dilation*(kernel_size - 1)//2
        self.layer = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,padding=padding,dilation = dilation),
                                   nn.GroupNorm(out_channels//4,out_channels),
                                   nn.PReLU(),
                                   nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,padding=padding,dilation = dilation),
                                   nn.GroupNorm(out_channels//4,out_channels),
                                   nn.PReLU())
    def forward(self,x) :
        return self.layer(x)

class InceptionLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_sizes = [1,3,5,7],dilation = 1) :
        super(InceptionLayer,self).__init__()
        intermed_channels = 2*out_channels
        self.layers = nn.ModuleList()
        for ksize in kernel_sizes :
            self.layers.append(DoubleConvLayer(in_channels,intermed_channels,ksize,dilation))
        self.output_layer = nn.Sequential(DoubleConvLayer(len(kernel_sizes)*intermed_channels,out_channels,1))
    def forward(self,x) :
        outputs = []
        for layer in self.layers :
            outputs.append(layer(x))
        output = torch.cat(outputs,dim=1)
        output = self.output_layer(output)
        return output

class ResidualLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,dilation = 1) :
        super(ResidualLayer,self).__init__()
        self.inp = DoubleConvLayer(in_channels,out_channels)
        self.residual_layer = InceptionLayer(out_channels,out_channels,dilation = dilation)
    def forward(self,x) :
        output = self.inp(x)
        output = self.residual_layer(output)+output
        return output