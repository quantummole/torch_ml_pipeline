# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 18:40:24 2018

@author: quantummole
"""

import random

class Search(object) :
    def __init__(self,max_configs) :
        self.configs = [i for i in range(max_configs)]
        self.total_configs =max_configs
    def get_next_state(self) :
        raise NotImplementedError
    def tune(self,val_score) :
        raise NotImplementedError

class GridSearch(Search) :
    def __init__(self,max_configs) :
        super(GridSearch,self).__init__(max_configs)
        self.curr_state = 0
    def get_next_state(self):
        value =  self.configs[self.curr_state]
        self.curr_state += 1
        if self.curr_state == self.total_configs :
            self.curr_state = 0
        return value
    def tune(self,val_score) :
        pass

class RandomSearch(Search) :
    def __init__(self,params_space,config_file,num_configs) :
        super(RandomSearch,self).__init__(params_space,config_file) 
        self.curr_count = self.total_configs
    def get_next_state(self,curr_state,max_grid) :
        if self.curr_count == self.total_configs :
            random.shuffle(self.configs)
        value = self.configs[self.curr_count - 1]
        self.curr_count -= 1
        if self.curr_count == 0 :
            self.curr_count = self.total_configs
        return value
    def tune(self,val_score) :
        pass