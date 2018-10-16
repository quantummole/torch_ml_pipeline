# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:41:52 2018

@author: quantummole
"""
import random

class BoolTransform :
    def __init__(self,transform) :
        self.transform = transform
        self.prob = 0.5
    def __call__(self,x) :
        if isinstance(x,list) :
            if random.random() < self.prob :
                return [self.transform(y) for y in x]
        else :
            if random.random() < self.prob :
                return self.transform(x)
        return x