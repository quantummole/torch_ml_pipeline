# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import numpy as np
import pandas as pd
from signals import Signal

class Fold(object):
    def __init__(self,training_split,evaluator = None):
        self.training_split = training_split
        self.validation_split = 1 - self.training_split
        self.num_folds = np.int(np.ceil(1.0/self.validation_split))
        self.is_init = True
    def execute(self,dataset):
        if self.is_init :
            self.is_init = False
            self.training_samples = np.int(self.training_split*dataset.shape[0])
            self.validation_samples = dataset.shape[0] - self.training_samples
            self.generate_folds(dataset)
        train_dataset,validation_dataset = self.get_fold(self.num_folds-1)
        self.num_folds -= 1
        completion_signal = Signal.COMPLETE if self.num_folds == 0 else Signal.INCOMPLETE
        return completion_signal,[train_dataset,validation_dataset] 
    def generate_folds(self,dataset) :
        raise NotImplementedError
    def get_fold(self,fold_num) :
        raise NotImplementedError

class ShuffleFold(Fold) :
    def __init__(self,training_split,evaluator = None) :
        super(ShuffleFold,self).__init__(training_split,evaluator) 
    def generate_folds(self,dataset) :
        self.dataset = dataset        
    def get_fold(self,fold_num) :
        train_data = self.dataset.sample(n=self.training_samples)
        validation_data = self.dataset.drop(train_data.index)
        return train_data,validation_data
    
    
class DeterministicFold(Fold) :
    def __init__(self,training_split,evaluator = None) :
        super(DeterministicFold,self).__init__(training_split,evaluator)    
    def generate_folds(self,dataset) :
        self.dataset = dataset
        self.folds = []
        curr_indices = []
        curr_dataset = self.dataset
        for i in range(0,self.dataset.shape[0],self.validation_samples) :
            pruned_dataset = curr_dataset.drop(curr_indices)
            if pruned_dataset.shape[0] == 0 :
                break
            self.folds.append(pruned_dataset.sample(n=min([pruned_dataset.shape[0],self.validation_samples])))
            curr_indices = self.folds[-1].index
            curr_dataset = pruned_dataset
        self.num_folds = len(self.folds)
    def get_fold(self,fold_num) :
        validation_data = self.folds[fold_num]
        train_data = self.dataset.drop(validation_data.index)
        return train_data,validation_data

class StratifiedDeterministicFold(Fold) :
    def __init__(self,training_split,group_keys,evaluator = None) :
        super(StratifiedDeterministicFold,self).__init__(training_split,evaluator)
        self.group_keys = group_keys
    def generate_folds(self,dataset) :
        self.dataset = dataset
        self.folds = []
        curr_indices = []
        curr_dataset = self.dataset
        dataset_groups = curr_dataset.groupby(self.group_keys,as_index=False,group_keys=False)
        dataset_group_names = dataset_groups.groups.keys()
        dataset_group_lengths = [len(dataset_groups.get_group(key)) for key in dataset_group_names]
        sample_group_lengths = [np.int(np.ceil(x*self.validation_split)) for x in dataset_group_lengths]
        for i in range(0,self.dataset.shape[0],self.validation_samples) :
            pruned_dataset = curr_dataset.drop(curr_indices)
            if pruned_dataset.shape[0] == 0 :
                break
            dataset_groups = pruned_dataset.groupby(self.group_keys,as_index=False,group_keys=False)
            dataset_group_names = dataset_groups.groups.keys()
            dataset_group_lengths = [len(dataset_groups.get_group(key)) for key in dataset_group_names]
            fold = []
            for i,key in enumerate(dataset_group_names) :
                fold.append(dataset_groups.get_group(key).sample(n=min([dataset_group_lengths[i],sample_group_lengths[i]])))
            fold = pd.concat(fold)
            self.folds.append(fold)
            curr_indices = self.folds[-1].index
            curr_dataset = pruned_dataset
        self.num_folds = len(self.folds)
    def get_fold(self,fold_num) :
        validation_data = self.folds[fold_num]
        train_data = self.dataset.drop(validation_data.index)
        return train_data,validation_data
    
class StratifiedShuffleFold(Fold) :
    def __init__(self,training_split,group_keys,evaluator = None) :
        super(StratifiedShuffleFold,self).__init__(training_split,evaluator)
        self.group_keys = group_keys
    def generate_folds(self,dataset) :
        self.dataset = dataset
    def get_fold(self,fold_num) :
        orig_index_length = self.dataset.index[0]
        orig_index_length = 1 if np.isscalar(orig_index_length) else len(orig_index_length)
        dataset_groups = self.dataset.groupby(self.group_keys)
        train_data = dataset_groups.apply(lambda x : x.sample(frac=self.training_split))
        new_index_length = len(train_data.index[0])
        ignore_length = new_index_length - orig_index_length
        validation_data = self.dataset.drop([x[ignore_length:] for x in train_data.index.values.tolist()])
        return train_data,validation_data