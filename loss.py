# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:05:55 2018

@author: quantummole
"""

class ClassificationLossList :
    def __init__(self,list_loss_fn,weights) :
        self.loss_fn = [loss_fn() for loss_fn in list_loss_fn]
        self.weights = weights
    def __call__(self,predictions,labels):
        loss = 0
        num_outputs = len(predictions)
        for i in range(len(self.loss_fn)) :
            for j in range(num_outputs) :
                loss +=  self.weights[i]*self.loss_fn[i](predictions[j],labels[j])/num_outputs
        return loss
 
class SiameseLossList :
    def __init__(self,list_loss_fn,weights) :
        self.loss_fn = [loss_fn() for loss_fn in list_loss_fn]
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
    