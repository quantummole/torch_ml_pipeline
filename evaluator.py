# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 08:01:34 2018

@author: quantummole
"""

class Evaluator:
    def __init__(self,log_dir,run_id,objective_fns) :
        self.run_id = run_id
        self.objective_fns = objective_fns
        self.log_dir = log_dir
        
    def log(self,mode,outputs,targets) :
        pass
    
    def get(self,mode) :
        return self.objective_fns[mode]
