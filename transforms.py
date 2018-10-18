# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:41:52 2018

@author: quantummole
"""
import random
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
class BoolTransform :
    def __init__(self,transform,prob = 0.5) :
        self.transform = transform
        self.prob = prob
    def __call__(self,x) :
        if isinstance(x,list) :
            if random.random() < self.prob :
                return [self.transform(y) for y in x]
        else :
            if random.random() < self.prob :
                return self.transform(x)
        return x

class PILTransform :
    def __init__(self,transform) :
        self.transform = transform
    def __call__(self,x) :
        if isinstance(x,list) :
            x = [np.array(self.transform(Image.fromarray(img_as_ubyte(y)))) for y in x]
            return x
        else :
            x = Image.fromarray(img_as_ubyte(x))
            x = self.transform(x)
            x = np.array(x)
            return x

def channel_first(im) :
    im = np.transpose(im,(2,0,1))
    return im

def hflip(x):
    return x[:,::-1]

def vflip(x):
    return x[::-1,:]

def transpose(x):
    dim = x.ndim
    remaining = [i+2 for i in range(dim-2)]
    return np.transpose(x,axes=[1,0]+remaining)