# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:05:55 2018

@author: quantummole
"""
import torch 

class ClassificationLossList :
    def __init__(self,list_list_loss_fn,weights) :
        self.loss_fn = list_list_loss_fn
        self.weights = weights
    def __call__(self,predictions,labels):
        loss = 0
        num_outputs = len(predictions)
        for i in range(num_outputs) :
            for j in range(len(self.loss_fn[i])) :
                loss +=  self.weights[i][j]*self.loss_fn[i][j](predictions[i],labels[i].type(torch.cuda.LongTensor))/num_outputs
        return loss
 
class SiameseLossList :
    def __init__(self,list_loss_fn,weights) :
        self.loss_fn = list_loss_fn
        self.weights = weights
    def __call__(self,predictions,labels):
        loss = 0
        num_classes = len(predictions)//2
        for i in range(len(self.loss_fn)) :
            for cls in range(num_classes) :
                positives = predictions[2*cls]
                anchors = predictions[2*cls+1]
                negatives = predictions[0:2*cls] + predictions[2*(cls+1):2*num_classes]
                loss +=  self.weights[i]*self.loss_fn[i](positives,anchors,negatives)/num_classes
        return loss    

class Accuracy :
    def __init__(self) : pass
    def __call__(self,predictions,labels) :
        _,vals = torch.max(predictions,dim=1)
        return 1.0 - torch.mean((vals.view(-1,1)==labels.view(-1,1)).type_as(predictions))

class MarginLoss :
    def __init__(self,num_classes) : 
        self.num_classes = num_classes
    def __call__(self,predictions,labels) :
        one_hot = torch.zeros_like(predictions)
        one_hot = one_hot.scatter(dim=1,index=labels.view(-1,1),source=1)
        values = torch.sum(one_hot*predictions,dim=1,keepdim=True)
        prediction_margins = predictions-values
        loss  = torch.mean(torch.log(torch.sum(torch.exp(prediction_margins),dim=1)))
        return loss