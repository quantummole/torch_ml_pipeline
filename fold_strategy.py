# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import random
import numpy as np
import pickle
import torch
from torch.utils import data


class Fold(object):
    def __init__(self,dataset,training_split):
        self.dataset = dataset
        self.training_split = training_split
        self.validation_split = 1 - self.training_split
        self.training_samples = np.int(self.training_split*self.dataset.shape[0])
        self.validation_samples = self.dataset.shape[0] - self.training_samples
        self.num_folds = np.int(np.ceil(1.0/self.validation_split))
    def __call__(self,fold_num):
        raise NotImplementedError


class ShuffleFold(Fold) :
    def __init__(self,dataset,training_split) :
        super(ShuffleFold,self).__init__(dataset,training_split)    
    def __call__(self,fold_num) :
        train_data = self.dataset.sample(n=self.training_samples)
        validation_data = self.dataset.drop(train_data.index)
        return train_data,validation_data
    
    
class DeterministicFold(Fold) :
    def __init__(self,dataset,training_split) :
        super(DeterministicFold,self).__init__(dataset,training_split)    
        self.folds = []
        curr_indices = []
        curr_dataset = self.dataset
        for i in range(0,self.dataset.shape[0],self.validation_samples) :
            pruned_dataset = curr_dataset.drop(curr_indices)
            self.folds.append(pruned_dataset.sample(n=min([pruned_dataset.shape[0],self.validation_samples])))
            curr_indices = self.folds[-1].index
            curr_dataset = pruned_dataset
        self.num_folds = len(self.folds)
    def __call__(self,fold_num) :
        validation_data = self.folds[fold_num]
        train_data = self.dataset.drop(validation_data.index)
        return train_data,validation_data

class ShuffleFoldGroups(Fold) :
    def __init__(self,dataset,training_split,group_keys) :
        super(ShuffleFoldGroups,self).__init__(dataset,training_split)
        self.group_keys = group_keys
    def __call__(self,fold_num) :
        dataset_groups = self.dataset.groupby(self.group_keys)
        train_data = dataset_groups.apply(lambda x : x.sample(frac=self.training_split))
        validation_data = self.dataset.drop([x[-1] for x in train_data.index.values.tolist()])
        return train_data,validation_data