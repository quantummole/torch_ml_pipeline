# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:05:55 2018

@author: quantummole
"""
import torch 
import torch.nn.functional as tfunc

class SupervisedMetricList :
    def __init__(self,list_list_objective_fn,weights) :
        self.objective_fn = list_list_objective_fn
        self.weights = weights
    def __repr__(self) :
        return repr(self.objective_fn)
    def __call__(self,predictions,labels):
        loss = 0
        num_outputs = len(predictions)
        for i in range(num_outputs) :
            for j in range(len(self.objective_fn[i])) :
                loss +=  self.weights[i][j]*self.objective_fn[i][j](predictions[i],labels[i])/num_outputs
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
    def __init__(self) :
        pass
    def __call__(self,logits,target) :
        target = target.view(-1,1)
        outputs_onehot = torch.zeros(logits.shape[0],logits.shape[1]).type_as(logits)
        outputs_onehot = outputs_onehot.scatter(1,target,1.0) 
        predictions = torch.sum(torch.exp(-1*outputs_onehot*logits),dim=1,keepdim=True)*torch.exp(logits) - outputs_onehot
        return torch.mean(torch.log(1 + predictions.mean(dim=1)))

class SoftDice :
    def __init__(self,num_classes,smooth=1) :
        self.num_classes = num_classes
        self.smooth = smooth
    def __call__(self,logits,mask) :
        predictions = tfunc.softmax(logits,dim=1)
        batch_size = predictions.shape[0]
        score = 0.
        for cls in range(self.num_classes) :
            probs = predictions[:,cls,:,:]
            probs = probs.view(batch_size,-1)
            label_cls = (mask == cls).view(batch_size,-1).type_as(probs)
            intersection = (probs*label_cls)
            union = probs + label_cls - intersection
            intersection = intersection.sum(dim=1) + self.smooth
            union = union.sum(dim=1) + self.smooth
            score = score + torch.mean((1 - intersection/union))
        return score

class DiceAccuracy :
    def __init__(self,num_classes,smooth=1) :
        self.num_classes = num_classes
        self.smooth = smooth
    def __call__(self,logits,mask) :
        predictions = tfunc.softmax(logits,dim=1).cpu()
        _,labels = torch.max(predictions,dim=1)
        batch_size = predictions.shape[0]
        score = 0.
        mask = mask.cpu()
        for cls in range(1,self.num_classes) :
            probs = (labels == cls).view(batch_size,-1).type_as(predictions)
            label_cls = (mask == cls).view(batch_size,-1).type_as(predictions)
            intersection = (probs*label_cls)
            union = probs + label_cls - intersection
            intersection = intersection.sum(dim=1) + self.smooth
            union = union.sum(dim=1) + self.smooth
            score = score + torch.mean((1 - intersection/union))
        return score.type_as(logits)