# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:52:38 2018

@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class DoubleConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_size = 3) :
        super(DoubleConvLayer,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,padding=kernel_size//2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,padding=kernel_size//2),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))
    def forward(self,x) :
        return self.layer(x)

class InceptionLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_sizes = [1,3,5]) :
        super(InceptionLayer,self).__init__()
        intermed_channels = 2*out_channels
        self.layers = nn.ModuleList()
        for ksize in kernel_sizes :
            self.layers.append(nn.Sequential(nn.Conv2d(in_channels,intermed_channels,kernel_size = ksize,padding=ksize//2),
                                            nn.BatchNorm2d(intermed_channels),
                                            nn.ReLU(inplace=True)))
        self.output_layer = nn.Sequential(nn.Conv2d(len(kernel_sizes)*intermed_channels,out_channels,kernel_size=1),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(inplace=True))
    def forward(self,x) :
        outputs = []
        for layer in self.layers :
            outputs.append(layer(x))
        output = torch.cat(outputs,dim=1)
        output = self.output_layer(output)
        return output
