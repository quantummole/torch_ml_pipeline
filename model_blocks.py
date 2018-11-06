# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:52:38 2018

@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc
import numpy as np
import random
class SingleConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_size = 3,dilation = 1, stride=1,padding= 0,activation = nn.PReLU, num_groups = None, noise_prob = 0.2) :
        super(SingleConvLayer,self).__init__()
        padding = dilation*(kernel_size - 1)//2
        self.num_groups = num_groups if num_groups else 8 if out_channels % 8 ==0 else 4 if out_channels % 4 ==0 else 2 if out_channels % 2 ==0 else 1 
        self.eps = 1e-5

        self.layer = nn.Sequential(nn.Conv2d(in_channels,out_channels,
                                             kernel_size = kernel_size,
                                             padding = padding,
                                             stride = stride,
                                             dilation = dilation),
                                   activation(),
                                   nn.GroupNorm(self.num_groups,out_channels))
        self.noise_prob = noise_prob                
    def forward(self,x) :
        output = self.layer(x)
        if self.training :
            if random.random() < self.noise_prob :
                output = output + 0.001*torch.randn_like(output)
        return output

class DoubleConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_size = 3,dilation = 1) :
        super(DoubleConvLayer,self).__init__()
        padding = dilation*(kernel_size - 1)//2
        self.layer = nn.Sequential(SingleConvLayer(in_channels,out_channels,kernel_size = kernel_size,padding=padding,dilation = dilation),
                                   SingleConvLayer(out_channels,out_channels,kernel_size = kernel_size,padding=padding,dilation = dilation))

    def forward(self,x) :
        return self.layer(x)

class InceptionLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,kernel_sizes = [1,3,5,7],dilation = 1) :
        super(InceptionLayer,self).__init__()
        intermed_channels = 2*out_channels
        self.layers = nn.ModuleList()
        for ksize in kernel_sizes :
            self.layers.append(DoubleConvLayer(in_channels,intermed_channels,ksize,dilation))
        self.output_layer = SingleConvLayer(len(kernel_sizes)*intermed_channels,out_channels,kernel_size=1)
    def forward(self,x) :
        outputs = []
        for layer in self.layers :
            outputs.append(layer(x))
        output = torch.cat(outputs,dim=1)
        output = self.output_layer(output)
        return output

class DenseBlock(nn.Module) :
    def __init__(self,in_channels,out_channels,dilation = 1,num_layers = 3) :
        super(DenseBlock,self).__init__()
        init_layers = in_channels
        self.dense_layers = nn.ModuleList()
        for i in range(num_layers) :
            self.dense_layers.append(InceptionLayer(init_layers,in_channels,dilation=dilation))
            init_layers += in_channels
        self.out_layer = SingleConvLayer(init_layers,out_channels,kernel_size=1)
    def forward(self,x) :
        for layer in self.dense_layers :
            output = layer(x)
            x = torch.cat([output,x],dim=1)
        output = self.out_layer(x)
        return output
        
class ResidualBlock(nn.Module) :
    def __init__(self,out_channels,dilation = 1,num_layers = 2) :
        super(ResidualBlock,self).__init__()
        self.bottle_neck = SingleConvLayer(out_channels,out_channels//4,kernel_size = 1)
        self.residual_layer = DenseBlock(out_channels//4,out_channels//4,dilation = dilation,num_layers = num_layers)
        self.expand_neck = SingleConvLayer(out_channels//4,out_channels,kernel_size=1)
    def forward(self,x) :
        output = self.bottle_neck(x)
        output = self.residual_layer(output)
        output = self.expand_neck(output)
        output = output + x
        return output

class HighwayBlock(nn.Module) :
    def __init__(self,out_channels,dilation = 1,num_layers = 2) :
        super(HighwayBlock,self).__init__()
        self.bottle_neck = SingleConvLayer(out_channels,out_channels//4,kernel_size=1)
        self.residual_layer = DenseBlock(out_channels//4,out_channels//4,dilation = dilation,num_layers = num_layers)
        self.expand_neck = SingleConvLayer(out_channels//4,out_channels,kernel_size = 1)
        self.highway_connection = nn.Sequential(nn.Conv2d(2*out_channels,out_channels,kernel_size=1),nn.Sigmoid())
    def forward(self,x) :
        output = self.bottle_neck(x)
        output = self.residual_layer(output)
        output = self.expand_neck(output)
        gate = self.highway_connection(torch.cat([output,x],dim=1))
        output = gate*output + (1-gate)*x
        return output

class RecConvLayer(nn.Module) :
    def __init__(self,in_channels,out_channels,num_groups) :
        super(RecConvLayer,self).__init__()
        self.conv_layer = DoubleConvLayer(in_channels,out_channels)
        self.horizontal_layer = nn.ModuleList() 
        self.vertical_layer = nn.ModuleList()
        self.horizontal1_layer = nn.ModuleList() 
        self.vertical1_layer = nn.ModuleList()
        self.num_groups = num_groups
        for i in range(self.num_groups) :
            self.horizontal1_layer.append(nn.GRU(input_size=out_channels,hidden_size=out_channels,num_layers=2,bidirectional=True))
            self.horizontal_layer.append(nn.GRU(input_size=out_channels,hidden_size=out_channels,num_layers=2,bidirectional=True))
            self.vertical1_layer.append(nn.GRU(input_size=out_channels,hidden_size=out_channels,num_layers=2,bidirectional=True))
            self.vertical_layer.append(nn.GRU(input_size=out_channels,hidden_size=out_channels,num_layers=2,bidirectional=True))
        self.final_comb = SingleConvLayer(2*out_channels,out_channels,kernel_size=1)
    def forward(self,x) :
        x = self.conv_layer(x)
        bs,c,m,n = x.shape
        y = torch.transpose(x,1,3)
        vert_groups = n//self.num_groups
        horizontal_groups = m//self.num_groups
        y1 = y.contiguous().view((bs,self.num_groups,m*vert_groups,c))
        y1 = torch.transpose(y1,0,2)
        y = torch.transpose(y,1,2)
        y2 = y.contiguous().view(bs,self.num_groups,n*horizontal_groups,c) 
        y2 = torch.transpose(y2,0,2)
        hidden_h = torch.zeros((2*2,bs,c)).type_as(x)
        h_n = []
        v_n = []
        for i in range(self.num_groups) :
            h_n1,_ = self.horizontal_layer[i](y2[:,i,:,:],hidden_h)
            v_n1,_ = self.vertical_layer[i](y1[:,i,:,:],hidden_h)
            h_n1 = torch.mean(h_n1.view(n*horizontal_groups,bs,2,c),dim=2)
            v_n1 = torch.mean(v_n1.view(m*vert_groups,bs,2,c),dim=2)
            h_n.append(h_n1)
            v_n.append(v_n1)
        h_n = torch.stack(h_n,dim=1)
        h_n = torch.transpose(h_n,0,2).contiguous().view(bs,m,n,c) 
        h_n = torch.transpose(h_n,1,2)
        y1 =  h_n.contiguous().view((bs,self.num_groups,m*vert_groups,c))
        v_n = torch.stack(v_n,dim=1)
        v_n = torch.transpose(v_n,0,2).contiguous().view(bs,n,m,c)
        v_n = torch.transpose(v_n,1,2)
        y2 = v_n.contiguous().view(bs,self.num_groups,n*horizontal_groups,c)
        y1 = torch.transpose(y1,0,2)
        y2 = torch.transpose(y2,0,2)
        h_n = []
        v_n = []
        for i in range(self.num_groups) :
            h_n1,_ = self.horizontal1_layer[i](y2[:,i,:,:],hidden_h)
            v_n1,_ = self.vertical1_layer[i](y1[:,i,:,:],hidden_h)
            h_n1 = torch.mean(h_n1.view(n*horizontal_groups,bs,2,c),dim=2)
            v_n1 = torch.mean(v_n1.view(m*vert_groups,bs,2,c),dim=2)
            h_n.append(h_n1)
            v_n.append(v_n1)
        h_n = torch.stack(h_n,dim=1)
        h_n = torch.transpose(h_n,0,2).contiguous().view(bs,m,n,c)
        h_n = torch.transpose(h_n,1,2)
        v_n = torch.stack(v_n,dim=1)
        v_n = torch.transpose(v_n,0,2).contiguous().view(bs,n,m,c)
        h_n = torch.transpose(h_n,1,3)
        v_n = torch.transpose(v_n,1,3)
        return self.final_comb(torch.cat([h_n,v_n],dim=1))