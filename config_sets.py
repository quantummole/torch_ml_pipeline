# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 07:42:33 2018

@author: quant
"""

from signals import Signal
from functools import reduce
from search_strategy import GridSearch
class ConfigSet :
    def __init__(self,params_space,searcher_class) :
        self.config_set = params_space
        self.completion_state =  Signal.INCOMPLETE
        self.curr_count = 0
        self.searcher_class = searcher_class
        self.searcher = None
    def __repr__(self) :
        return self.__class__.__name__+"("+repr(self.config_set)+")"
    def initialize_searcher(self) :
        self.searcher = self.searcher_class(self.get_max_configs())
    def get_next(self,score) :
        if self.searcher == None :
            self.initialize_searcher()
        if not Signal.NO_SCORE :
            self.searcher.tune(score)
        config_num = self.searcher.get_next_state()
        config = self.get_config(config_num)
        self.curr_count += 1
        if self.curr_count == self.get_max_configs() :
            self.completion_state = Signal.COMPLETE
        return config_num,config
    def get_max_configs(self) :
        raise   NotImplementedError
    def get_config(self,config_num) :
        raise NotImplementedError
    def reinitialize(self) : 
         raise NotImplementedError    

class ExclusiveConfigs(ConfigSet) :
    def __init__(self,params_space,searcher=GridSearch,max_configs = None):
        super(ExclusiveConfigs,self).__init__(params_space,searcher)
        self.param_grid = []
        for value in self.config_set :
            if isinstance(value,ConfigSet):
                self.param_grid.append(value.max_configs)
            else :
                self.param_grid.append(1)
        self.max_configs = max_configs if max_configs else reduce(lambda x,y : x+y,self.param_grid)
    def get_max_configs(self) :
        return self.max_configs
    def get_config(self,config_num) :
        for i,value in enumerate(self.param_grid) :
            if config_num <= value :
                item = self.config_set[i]
                break
            else :
                config_num -= value
        if isinstance(item,ConfigSet) :
            if item.completion_state == Signal.COMPLETE :
                item.reinitialize()
            item = item.get_config(config_num)
        return item
    def reinitialize(self) :
        self.completion_state = Signal.INCOMPLETE
        self.curr_count = 0
        for item in self.config_set :
            if isinstance(item,ConfigSet) :
                item.reinitialize()

class NamedConfig(ConfigSet) :
    def __init__(self,params_space,searcher=GridSearch,max_configs = None):
        super(NamedConfig,self).__init__(params_space,searcher)   
        self.param_grid = []
        key,value = self.config_set
        if isinstance(value,ConfigSet):
            self.param_grid.append(value.get_max_configs())
        else :
            self.param_grid.append(1)
        self.max_configs = max_configs if max_configs else  self.param_grid[0]
    def get_max_configs(self) :
        return self.max_configs
    def get_config(self,config_num) :
        key = self.config_set[0]
        item = self.config_set[1]
        if isinstance(item,ConfigSet) :
            if item.completion_state == Signal.COMPLETE :
                item.reinitialize()
            item = item.get_config(config_num)
        return (key,item)
    def reinitialize(self) :
        self.completion_state = Signal.INCOMPLETE
        self.curr_count = 0
        for key,item in self.config_set :
            if isinstance(item,ConfigSet) :
                item.reinitialize()

class DictConfig(ConfigSet) :
    def __init__(self,params_space,searcher=GridSearch,max_configs = None):
        super(DictConfig,self).__init__(params_space,searcher)   
        self.param_grid = []
        for value in self.config_set :
            self.param_grid.append(value.max_configs)
        self.max_configs = max_configs if max_configs else reduce(lambda x,y : x*y,self.param_grid)
    def get_max_configs(self) :
        return self.max_configs
    def get_config(self,config_num) :
        config_num_list = [1]*len(self.param_grid)
        for i,value in enumerate(self.param_grid) :
            if config_num <= value :
                config_num_list[i] = config_num
                break
            else :
                config_num_list[i] = (config_num % value) + 1
                config_num = config_num//value
        configs = [self.config_set[i].get_config(config_num) for i,config_num in enumerate(config_num_list)]
        return dict(configs)
                
    def reinitialize(self) :
        self.completion_state = Signal.INCOMPLETE
        self.curr_count = 0
        for config in self.config_set :
            config.reinitialize()

class CombinerConfig(ConfigSet) :
    def __init__(self,params_space,searcher=GridSearch,max_configs = None):
        super(CombinerConfig,self).__init__(params_space,searcher)   
        self.param_grid = []
        for value in self.config_set :
            self.param_grid.append(value.max_configs)
        self.max_configs = max_configs if max_configs else reduce(lambda x,y : x*y,self.param_grid)
    def get_max_configs(self) :
        return self.max_configs
    def get_config(self,config_num) :
        config_num_list = [1]*len(self.param_grid)
        for i,value in enumerate(self.param_grid) :
            if config_num <= value :
                config_num_list[i] = config_num
                break
            else :
                config_num_list[i] = (config_num % value) + 1
                config_num = config_num//value
        configs = {}
        for i,config_num in enumerate(config_num_list) :
            config = self.config_set[i].get_config(config_num)
            configs = {**configs,**config}
        return configs
                
    def reinitialize(self) :
        self.completion_state = Signal.INCOMPLETE
        self.curr_count = 0
        for config in self.config_set :
            config.reinitialize()