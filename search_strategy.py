# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:40:24 2018

@author: quantummole
"""

import pickle
import hashlib
import random
from functools import reduce 
class Search(object) :
    def __init__(self,params_space,config_file) :
        self.params_space = params_space
        self.config_file = config_file
        self.param_grid  = [len(self.params_space[key][value]) if isinstance(self.params_space[key][value],list) else 1 for key in self.params_space.keys() for value in self.params_space[key].keys()]
        self.max_configs = reduce(lambda x, y: x*y, self.param_grid)
        self.curr_state = [0  for l  in self.param_grid]
        self.curr_state[0] = -1

    def generate_config_id(self,params) :
        md5 = hashlib.md5()
        md5.update(pickle.dumps(params))
        md5.update(pickle.dumps(random.random()))
        return md5.hexdigest()
    def store_config(self,config_id,params) :
        print("generating configuration {}".format(config_id),flush=True)
        file = open(self.config_file.format(config_id),"wb")
        pickle.dump(params,file)
        file.close()
    def get_next_state(self,curr_state,max_grid) :
        raise NotImplementedError
    def __call__(self) :
        self.curr_state = self.get_next_state(self.curr_state,self.param_grid)
        counter = 0
        params = {}
        for method in self.params_space.keys() :
            params[method] = {}
            for key in self.params_space[method].keys():
                values = self.params_space[method][key]
                value = -1
                if isinstance(values, list) :
                    value = values[self.curr_state[counter]]
                else : 
                    value = values
                params[method][key] = value
                counter += 1
        config_id = self.generate_config_id(params)
        self.store_config(config_id,params)
        return config_id,params
    def tune(self,val_score) :
        raise NotImplementedError

class GridSearch(Search) :
    def __init__(self,params_space,config_file) :
        super(GridSearch,self).__init__(params_space,config_file)
        self.param_keys = self.params_space.keys()
        self.param_names_grid = [[value for value in self.params_space[key]] for key in self.param_keys] 
    def get_next_state(self,curr_state,max_grid):
        if curr_state == []:
            return []
        else :
            if curr_state[0] < max_grid[0]-1 :
                return [curr_state[0]+1] + curr_state[1:]
            else :
                return [0] + self.get_next_state(curr_state[1:],max_grid[1:])
    def tune(self,val_score) :
        pass

class RandomSearch(Search) :
    def __init__(self,params_space,config_file) :
        super(RandomSearch,self).__init__(params_space,config_file) 
    def get_next_state(self,curr_state,max_grid) :
        return [random.randint(0,i-1) for i in max_grid]
    def tune(self,val_score) :
        pass