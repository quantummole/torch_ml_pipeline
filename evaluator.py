# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 08:01:34 2018

@author: quantummole
"""
import pickle
import hashlib

class Evaluator:
    def __init__(self,root_dir,inference_fn = None,debug_fn = None, logger_fn = None) :
        self.root_dir = root_dir
        self.model_dir = self.root_dir+"/models"
        self.model_file = self.model_dir+"/model_{}.mod"
        self.debug_dir = self.root_dir+"/debug/"
        self.config_dir = self.root_dir+"/configs"
        self.config_file = self.config_dir+"/config_{}.pkl"
        self.inference_dir = self.root_dir+"/inference/"
        self.inference_file = self.inference_dir+"/{}"
        self.metrics_dir = self.root_dir+"/metrics/"
        self.score_file = self.root_dir+"/score.pkl"
        self.inference_fn = None
        self.curr_config = {}
    def __repr__(self) :
        return self.__class__.__name__+"("+self.root_dir+")"
    def get_config_id(self) :
        md5 = hashlib.md5()
        md5.update(pickle.dumps(self.curr_config))
        return md5.hexdigest()
    def set_config(self,params) :
        self.curr_config = params
    def update_config(self,params) :
        self.curr_config = {**self.curr_config,**params}
    def get_model_file(self) :
        return self.model_file.format(self.get_config_id())
    def get_inference_file(self) :
        return self.inference_file.format(self.get_config_id())
    def log_config(self) :
        file = open(self.config_file.format(self.get_config_id()),"wb")
        pickle.dump(self.curr_config,file)
        file.close()
    def log(self,mode,outputs,targets,debug_info) :
        if mode == "inference" :
            self.inference_fn(self.inference_file,outputs,targets,debug_info)
    def log_score(self,run_id,score) :
        pass
