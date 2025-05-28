# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:05:55 2018

@author: quantummole
"""
import torch 
import torch.nn as nn
import torch.nn.functional as tfunc

class SupervisedLossList(nn.Module) :
    def __init__(self,list_list_objective_fn,weights) :
        super(SupervisedLossList,self).__init__()
        self.objective_fn = nn.ModuleList()
        for l in list_list_objective_fn :
            self.objective_fn.append(nn.ModuleList(modules=l))
        self.weights = weights
    def __repr__(self) :
        return repr(self.objective_fn)
    def forward(self,predictions,labels):
        loss = 0
        num_outputs = len(predictions)
        for i in range(num_outputs) :
            for j in range(len(self.objective_fn[i])) :
                loss +=  self.weights[i][j]*self.objective_fn[i][j](predictions[i],labels[i])/num_outputs
        return loss
 
class SiameseLossList(nn.Module) :
    def __init__(self,list_loss_fn,weights) :
        super(SiameseLossList,self).__init__()
        self.loss_fn = nn.ModuleList(modules=list_loss_fn)
        self.weights = weights
    def forward(self,predictions,labels):
        loss = 0
        num_classes = len(predictions)//2
        for i in range(len(self.loss_fn)) :
            for cls in range(num_classes) :
                positives = predictions[2*cls]
                anchors = predictions[2*cls+1]
                negatives = predictions[0:2*cls] + predictions[2*(cls+1):2*num_classes]
                loss +=  self.weights[i]*self.loss_fn[i](positives,anchors,negatives)/num_classes
        return loss    

class Accuracy(nn.Module) :
    def __init__(self) :
        super(Accuracy,self).__init__()
    def forward(self,predictions,labels) :
        _,vals = torch.max(predictions,dim=1)
        return 1.0 - torch.mean((vals.view(-1,1)==labels.view(-1,1)).type_as(predictions))