# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 08:01:34 2018

@author: quantummole
"""
import pickle
class Evaluator:
    def __init__(self,log_dir) :
        self.model_dir = log_dir+"/models"
        self.debug_dir = log_dir+"/debug/"
        self.config_dir = log_dir+"/configs"
        self.config_file = self.config_dir+"/config_{}.pkl"
        self.inference_dir = log_dir+"/inference/"
        self.metrics_dir = log_dir+"/metrics/"
        self.score_file = log_dir+"/score.pkl"
        
    def log(self,mode,outputs,targets,debug_info) :
        pass
    def log_config(self,config_id,config) :
        file = open(self.config_file.format(config_id),"wb")
        self.model_file = self.model_dir+"/model_{}.mod".format(config_id)
        pickle.dump(config,file)
        file.close()
