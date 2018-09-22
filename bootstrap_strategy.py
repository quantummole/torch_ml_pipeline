# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:09:23 2018

@author: quantummole
"""
import numpy as np
from signals import Signal
class Sample :
    def __init__(self,num_samples) :
        self.num_samples = num_samples
    def execute(self,dataset) :
        self.dataset = dataset
        subset = self.sample(self.num_samples)
        self.num_samples -= 1
        completion_state = Signal.COMPLETE if self.num_samples == 0 else Signal.INCOMPLETE
        return completion_state,str(self.num_samples),[subset]
    def sample(self,sample_id) :
        raise NotImplementedError

class NoBootStrap(Sample) :
    def __init__(self) :
        super(NoBootStrap,self).__init__(1)
    def sample(self,sample_id) :
        return self.dataset

class UnderSample(Sample) :
    def __init__(self,num_samples,group_keys,replace=False) :
        super(UnderSample,self).__init__(num_samples)
        self.group_keys = group_keys
        self.replace = replace
    def sample(self,sample_id) :
        dataset_groups = self.dataset.groupby(self.group_keys,as_index=False)
        min_samples = np.min(dataset_groups.apply(lambda x : len(x)).values)
        subset = dataset_groups.apply(lambda x : x.sample(n=min_samples,replace=self.replace))
        return subset

class OverSample(Sample) :
    def __init__(self,num_samples,group_keys,fraction) :
        super(OverSample,self).__init__(num_samples)
        self.group_keys = group_keys
        self.fraction = fraction
    def sample(self,sample_id) :
        dataset_groups = self.dataset.groupby(self.group_keys,as_index=False)
        max_samples = np.max(dataset_groups.apply(lambda x : len(x)).values)
        required_fraction = np.int(np.ceil(self.fraction*max_samples))
        subset = dataset_groups.apply(lambda x : x.sample(n=np.max([required_fraction,len(x)]),replace= True if len(x) < required_fraction else False))
        return subset