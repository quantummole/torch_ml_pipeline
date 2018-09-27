# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:21:33 2018

@author: quantummole
"""
from signals import Signal
class Rename :
    def __init__(self,input_keys) :
        self.input_keys = input_keys
    def execute(self,**kwargs) :
        outputs = []
        for key in self.input_keys :
            outputs.append(kwargs[key])
        return Signal.COMPLETE,outputs
