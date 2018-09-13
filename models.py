# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:14:19 2018

@author: quantummole
"""
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from model_blocks import DoubleConvLayer

class create_net :
    def __init__(self,net) :
        self.net = net
    def __call__(self,network_params,weights = None) :
        network = self.net(**network_params)
        if weights :
            network.load_state_dict(torch.load(weights,map_location=lambda storage, loc: storage))
        return network
    
class CustomNet1(nn.Module):
    def __init__(self,input_dim,initial_channels,growth_factor,num_classes) :
        super(CustomNet1,self).__init__()
        self.layer = nn.ModuleList()
        while input_dim >= 8 :
            self.layer.append(nn.Sequential(DoubleConvLayer(initial_channels,initial_channels+growth_factor,3,1),
                                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)))
            input_dim = input_dim//2
            initial_channels += growth_factor
        self.output_layer = nn.Linear(input_dim*input_dim*initial_channels,num_classes)
    
    def forward(self,inputs,debug=False) :
        inputs = inputs[0]
        if len(inputs.shape) == 3 :
            bs,m,n = inputs.shape
            inputs = inputs.view(bs,1,m,n)
        bs,c,m,n = inputs.shape
        for layer in self.layer :
            inputs = layer(inputs)
        inputs = inputs.view(bs,-1)    
        output = self.output_layer(inputs)
        return [output]

class CustomNet2(nn.Module):
    def __init__(self,input_dim,initial_channels,growth_factor,embedding_size) :
        super(CustomNet2,self).__init__()
        self.layer = nn.ModuleList()
        while input_dim >= 8 :
            self.layer.append(nn.Sequential(DoubleConvLayer(initial_channels,initial_channels+growth_factor,3,1),
                                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)))
            input_dim = input_dim//2
            initial_channels += growth_factor
        self.output_layer = nn.Sequential(nn.Linear(input_dim*input_dim*initial_channels,embedding_size),nn.Tanh())
    
    def forward(self,inputs,debug=False) :
        outputs = []
        for inp in inputs :
            if len(inp.shape) == 3 :
                bs,m,n = inp.shape
                inp = inp.view(bs,1,m,n)
            bs,c,m,n = inp.shape
            for layer in self.layer :
                inp = layer(inp)
            inp = inp.view(bs,-1)    
            output = self.output_layer(inp)
            outputs.append(output)
        return outputs