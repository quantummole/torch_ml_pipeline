# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 20:09:23 2018

@author: quantummole
"""
import numpy as np

class Sample :
    def __init__(self,dataset,num_samples) :
        self.dataset = dataset
        self.num_samples = num_samples
    def sample(self,sample_id) :
        raise NotImplementedError

class NoBootStrap(Sample) :
    def __init__(self,dataset) :
        super(NoBootStrap,self).__init__(dataset,1)
    def sample(self,sample_id) :
        return self.dataset

class UnderSample(Sample) :
    def __init__(self,dataset,num_samples,group_keys,replace=False) :
        super(UnderSample,self).__init__(dataset,num_samples)
        self.group_keys = group_keys
        self.replace = replace
    def sample(self,sample_id) :
        dataset_groups = self.dataset.groupby(self.group_keys,as_index=False)
        min_samples = np.min(dataset_groups.apply(lambda x : len(x)).values)
        subset = dataset_groups.apply(lambda x : x.sample(n=min_samples,replace=self.replace))
        return subset

class OverSample(Sample) :
    def __init__(self,dataset,num_samples,group_keys,fraction) :
        super(OverSample,self).__init__(dataset,num_samples)
        self.group_keys = group_keys
        self.fraction = fraction
    def sample(self,sample_id) :
        dataset_groups = self.dataset.groupby(self.group_keys,as_index=False)
        max_samples = np.max(dataset_groups.apply(lambda x : len(x)).values)
        required_fraction = np.int(np.ceil(self.fraction*max_samples))
        subset = dataset_groups.apply(lambda x : x.sample(n=np.max([required_fraction,len(x)]),replace= True if len(x) < required_fraction else False))
        return subset