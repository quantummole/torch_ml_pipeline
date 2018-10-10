# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:05:55 2018

@author: quantummole
"""
import torch 
import torch.nn as nn
import torch.nn.functional as tfunc

class SupervisedMetricList(nn.Module) :
    def __init__(self,list_list_objective_fn,weights) :
        super(SupervisedMetricList,self).__init__()
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

class MultiLabelBCE(nn.Module) :
    def __init__(self) :
        super(MultiLabelBCE,self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self,logits,target) :
        target = target.view(-1,1)
        outputs_onehot = torch.zeros(logits.shape[0],logits.shape[1]).type_as(logits)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0).type_as(logits)
        loss = self.loss_fn(logits,outputs_onehot)
        return loss

class MarginLoss(nn.Module) :
    def __init__(self) :
        super(MarginLoss,self).__init__()
    def forward(self,logits,target) :
        target = target.view(logits.shape[0],1,-1)
        logits = logits/logits.norm(dim=1,keepdim=True)
        logits = logits.view(logits.shape[0],logits.shape[1],-1)
        outputs_onehot = torch.zeros_like(logits)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0) 
        predictions = (torch.exp(logits*(1 - outputs_onehot)) - outputs_onehot).mean(dim=1)*logits.shape[1]/(logits.shape[1]-1)
        predictions =  predictions*(torch.sum(torch.exp(-1*logits)*outputs_onehot,dim=1))
        return torch.mean(torch.log(1 + predictions))

class FocalCELoss(nn.Module) :
    def __init__(self,low=0.01,high=1.2,gamma = 2) :
        super(FocalCELoss,self).__init__()
        self.low = low
        self.high = high
        self.gamma = gamma
    def forward(self,logits,target) :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)
        outputs_onehot = torch.zeros_like(logits)
        target = target.view(logits.shape[0],1,-1)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0) 
        predictions = tfunc.softmax(logits,dim=1)
        factor = torch.clamp((1- predictions)/(predictions+ 1e-5),self.low,self.high).pow(self.gamma)
        loss = (-tfunc.log_softmax(logits,dim=1)*factor*outputs_onehot).sum(dim=1)
        return torch.mean(loss)

class SoftDice(nn.Module) :
    def __init__(self,smooth=1,low=0.01,high=1.2,gamma = 2) :
        super(SoftDice,self).__init__()
        self.smooth = smooth
        self.low = low
        self.high = high
        self.gamma = gamma
    def forward(self,logits,target) :
        predictions = tfunc.softmax(logits,dim=1)
        target = target.view(logits.shape[0],1,-1)
        predictions = predictions.view(logits.shape[0],logits.shape[1],-1)
        outputs_onehot = torch.zeros_like(predictions)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0)
        factor = torch.clamp((1- predictions)/(predictions+ 1e-5),self.low,self.high).pow(self.gamma)
        intersection = (factor*predictions*outputs_onehot).sum(dim=2)
        union = (factor*(predictions + outputs_onehot)).sum(dim=2) - intersection
        loss = 1 - (intersection+self.smooth)/(union+self.smooth)
        return torch.mean(loss)

class DiceAccuracy(nn.Module) :
    def __init__(self,smooth=1) :
        super(DiceAccuracy,self).__init__()
        self.smooth = smooth
    def forward(self,logits,target) :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)
        _,predictions = torch.max(logits,dim=1)
        outputs_onehot = torch.zeros_like(logits)
        target = target.view(logits.shape[0],1,-1)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0)
        predictions = predictions.view(logits.shape[0],1,-1)
        predictions_onehot = torch.zeros_like(logits)
        predictions_onehot = predictions_onehot.scatter(1,predictions,1.0)
        intersection = (predictions_onehot*outputs_onehot).sum(dim=2)
        union = (predictions_onehot + outputs_onehot).sum(dim=2) - intersection
        intersection = intersection.type_as(logits)
        union = union.type_as(logits)
        loss = 1 - (intersection+self.smooth)/(union+self.smooth)
        return torch.mean(loss)