# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:40:24 2018

@author: quantummole
"""

import pickle
import hashlib
import random

class Search(object) :
    def __init__(self,params_space,config_file) :
        self.params_space = params_space
        self.config_file = config_file
    def generate_config_id(self,params) :
        return hashlib.md5(pickle.dumps(params)).hexdigest()
    def store_config(self,config_id,params) :
        print("generating configuration {}".format(config_id),flush=True)
        file = open(self.config_file.format(config_id),"wb")
        pickle.dump(params,file)
        file.close()
    def __call__(self) :
        raise NotImplementedError
    def tune(self,val_score) :
        raise NotImplementedError

class RandomSearch(Search) :
    def __init__(self,params_space,config_file) :
        super(RandomSearch,self).__init__(params_space,config_file) 
    def __call__(self) :
        params = {}
        for method in self.params_space.keys() :
            params[method] = {}
            for key in self.params_space[method].keys():
                values = self.params_space[method][key]
                value = -1
                if isinstance(values, list) :
                    random.shuffle(values)
                    value = values[0]
                else : 
                    value = values
                params[method][key] = value
        config_id = self.generate_config_id(params)
        self.store_config(config_id,params)
        return config_id,params