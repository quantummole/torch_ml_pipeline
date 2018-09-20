# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:14:19 2018

@author: quantummole
"""
import torch
import torch.nn as nn
import torch.nn.functional as tfunc
from torchvision import models
from model_blocks import DoubleConvLayer

#mode = -1 is for debug
#mode = 0 is for test and validation
#mode = {1,2,3..} is for training

class create_net :
    def __init__(self,net) :
        self.net = net
    def __call__(self,network_params,weights = None) :
        network = nn.DataParallel(self.net(**network_params))
        if weights :
            network.load_state_dict(torch.load(weights,map_location=lambda storage, loc: storage))
        return network

class DensenetModels(nn.Module) :
    def __init__(self,model) :
        super(DensenetModels,self).__init__()
        self.model = model(pretrained=True)
        self.final_layer_features = self.model.classifier.in_features
    def update_final_layer(self,final_layer) :
        self.model.classifier = final_layer
    def forward(self,inputs,mode) :
        outputs = []
        for inp in inputs :
            if len(inp.shape) == 3 :
                bs,m,n = inp.shape
                inp = torch.stack([inp,inp,inp],dim=1)
            outputs.append(self.model(inputs))
        return outputs


class ResnetModels(nn.Module) :
    def __init__(self,model) :
        super(ResnetModels,self).__init__()
        self.model = model(pretrained=True)
        self.final_layer_features = self.model.fc.in_features
    def update_final_layer(self,final_layer) :
        self.model.fc = final_layer
    def forward(self,inputs,mode) :
        outputs = []
        for inp in inputs :
            if len(inp.shape) == 3 :
                bs,m,n = inp.shape
                inp = torch.stack([inp,inp,inp],dim=1)
            outputs.append(self.model(inputs))
        return outputs

class PreTrainedClassifier(nn.Module) :
    def __init__(self,model,num_classes) :
        super(PreTrainedClassifier,self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.model.update_final_layer(nn.Linear(self.model.final_layer_features,self.num_classes))
    def forward(self,inputs,mode) :
        return self.model(inputs,mode)
   
class CustomNetClassification(nn.Module):
    def __init__(self,input_dim, final_conv_dim, initial_channels,growth_factor,num_classes,conv_module) :
        super(CustomNetClassification,self).__init__()
        self.layer = nn.ModuleList()
        while input_dim >= final_conv_dim :
            self.layer.append(nn.Sequential(conv_module(initial_channels,initial_channels+growth_factor),
                                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)))
            input_dim = input_dim//2
            initial_channels += growth_factor
        num_units = input_dim*input_dim*initial_channels
        self.output_layer = nn.Sequential(nn.Linear(num_units,2*num_units),nn.ReLU(),
                                          nn.Linear(2*num_units,2*num_units),nn.ReLU(),
                                          nn.Linear(2*num_units,num_classes))
    
    def forward(self,inputs,mode) :
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