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
        self.model = model(pretrained=True)
        self.final_layer_features = self.model.classifier.in_features
    def update_final_layer(self,final_layer) :
        self.model.classifier = final_layer
    def forward(self,inputs) :
        if len(inputs.shape) == 3 :
            bs,m,n = inputs.shape
            inputs = inputs.view(bs,1,m,n)
        return self.model(inputs)


class ResnetModels(nn.Module) :
    def __init__(self,model) :
        self.model = model(pretrained=True)
        self.final_layer_features = self.model.fc.in_features
    def update_final_layer(self,final_layer) :
        self.model.fc = final_layer
    def forward(self,inputs) :
        if len(inputs.shape) == 3 :
            bs,m,n = inputs.shape
            inputs = inputs.view(bs,1,m,n)
        return self.model(inputs)

    
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
    
    def forward(self,inputs,mode=-1,debug=False) :
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
    
    def forward(self,inputs,mode=-1,debug=False) :
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

class CustomNet3(nn.Module):
    def __init__(self,num_classes_seg,num_classes_classify,hidden_size) :
        super(CustomNet3,self).__init__()
        self.layer = nn.ModuleList()
        self.num_classes = num_classes_seg
        self.num_classes_classifier = num_classes_classify
        self.hidden_size = hidden_size

        while input_dim >= 8 :
            self.layer.append(nn.Sequential(DoubleConvLayer(initial_channels,initial_channels+growth_factor,3,1),
                                            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)))
            input_dim = input_dim//2
            initial_channels += growth_factor
        self.final_size = input_dim*input_dim*initial_channels
        self.output_layer = nn.Linear(self.final_size,self.num_classes)
        self.rnn = nn.GRU(self.final_size,self.hidden_size,2,bidirectional=True)
        self.output_classifier = nn.Linear(2*self.hidden_size,self.num_classes_classifier)
        
    def forward(self,inputs,mode=-1,debug=False) :
        outputs = []
        inputs = inputs[0]
        if len(inputs.shape) == 3 :
            bs,m,n = inputs.shape
            inputs = inputs.view(bs,1,m,n)
        bs,c,m,n = inputs.shape
        init_hiddden = torch.zeros(4,bs,self.hidden_size).type_as(inputs)
        embeddings = []
        for i in range(0,m,32) :
            for j in range(0,n,32) :
                inp = inputs[:,:,i:i+32,j:j+32]
                for layer in self.layer :
                    inp = layer(inp)
                inp = inp.view(bs,-1)
                embeddings.append(inp)
                output = self.output_layer(inp)
                outputs.append(output)
        outputs = torch.stack(outputs,dim=2).view(bs,self.num_classes,m//32,n//32)
        rnn_inp = torch.stack(embeddings,dim=0)
        rnn_output,_ = self.rnn(rnn_inp,init_hiddden)
        rnn_output =  torch.mean(rnn_output,dim=0)
        classification_output = self.output_classifier(rnn_output)
        return [outputs,classification_output]

class CustomNet4(nn.Module):
    def __init__(self,num_classes_classification,num_classes_seg,output_dim,model_class) :
        super(CustomNet4,self).__init__()
        self.num_classes = num_classes_classification
        self.num_classes_seg = num_classes_seg
        self.output_dim = output_dim
        self.model = model_class
        self.model.update_final_layer(nn.Linear(self.final_layer_features,self.num_classes+self.output_dim*self.output_dim*self.num_classes_seg))
    def forward(self,inputs,mode=-1,debug=False) :
        inputs = inputs[0]
        if len(inputs.shape) == 3 :
            bs,m,n = inputs.shape
            inputs = torch.stack([inputs,inputs,inputs],dim=1)
        bs,c,m,n = inputs.shape
        output = self.model(inputs)
        classification_output = output[:,0:self.num_classes]
        if mode == 0  and self.training:
            return [classification_output]
        else :
            outputs = output[:,self.num_classes:].view(bs,self.num_classes_seg,self.output_dim,self.output_dim)
            return [classification_output,outputs]