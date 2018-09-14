# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import random
import numpy as np
import pickle
import torch
from torch.utils import data


class Fold(object):
    def __init__(self,dataset,training_samples,validation_samples):
        pass

    def __call__(self,fold_num):
        raise NotImplementedError


class ShuffleFold(Fold) :
    def __init__(self,dataset,training_samples,validation_samples) :
        self.dataset = dataset
        self.training_samples = training_samples
        self.validation_samples = validation_samples
    
    def __call__(self,fold_num) :
        train_data = self.dataset.sample(n=self.training_samples)
        validation_data = self.dataset.drop(train_data.index)
        return train_data,validation_data
    
    
class DeterministicFold(Fold) :
    def __init__(self,dataset,training_samples,validation_samples) :
        self.dataset = dataset
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.folds = []
        curr_indices = []
        curr_dataset = self.dataset
        for i in range(0,training_samples+validation_samples,validation_samples) :
            pruned_dataset = curr_dataset.drop(curr_indices)
            self.folds.append(pruned_dataset.sample(n=min([pruned_dataset.shape[0],self.validation_samples])))
            curr_indices = self.folds[-1].index
            curr_dataset = pruned_dataset
            
    def __call__(self,fold_num) :
        validation_data = self.folds[fold_num]
        train_data = self.dataset.drop(validation_data.index)
        return train_data,validation_data


class CrossValidation :
    def __init__(self,config_params,params_space,debug_mode=False) :
        self.network = config_params["network"]
        self.optimizer = config_params["optimizer"]
        self.scheduler = config_params["scheduler"]
        self.model_dir = config_params["model_dir"]+"/models"
        self.model_file = self.model_dir+"/model_{}_{}.mod"
        self.debug_dir = config_params["model_dir"]+"/debug/"
        self.device = config_params["device"]
        self.dataset = config_params["dataset"]
        self.config_dir = config_params["model_dir"]+"/configs"
        self.config_file = self.config_dir+"/config_{}.pkl"
        self.score_file = config_params["model_dir"]+"/score.pkl"
        self.params_space = params_space
        self.debug_mode = debug_mode
        self.fold_strategy = config_params["fold_strategy"]
        
    def RandomSearch(self) : 
        config_id = np.int(10000*random.random())
        print("generating configuration {}".format(config_id),flush=True)
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
        file = open(self.config_file.format(config_id), 'wb')
        pickle.dump(params,file)
        file.close()
        return config_id,params
    
    def get_params(self,config_id) :
        file = open(self.config_file.format(config_id), 'rb')
        params = pickle.load(file)
        file.close()
        return params
    
    def save_scores(self,config_scores) :
        file = open(self.score_file, 'wb')
        pickle.dump(config_scores,file)
        file.close()  
        
    def get_scores(self) :
        file = open(self.score_file, 'rb')
        scores = pickle.load(file)
        file.close()
        return scores
    
    def cross_validate(self,dataset,max_epochs,num_configs,tune_fn) :
        total_samples = dataset.shape[0]
        config_scores = {}
        for config_batch in range(num_configs) :
            config_id,params = tune_fn()
            training_split = params["data"]["training_split"]
            validation_split = 1 - training_split
            training_samples = np.int(training_split*total_samples)
            validation_samples = total_samples - training_samples
            sampler = self.fold_strategy(dataset,training_samples,validation_samples)
            num_folds = np.int(np.ceil(1.0/validation_split))
            validation = []
            training = []
            print("beginning cross validation",flush=True)
            for fold in range(num_folds) :
                fold_validation = []
                fold_training = []
                net = self.network(params["network"]).to(self.device)
                optimizer = self.optimizer(net.parameters(),**params["optimizer"])
                scheduler = self.scheduler(optimizer,**params["scheduler"])
                curr_best = params["constants"]["val_best"]
                train_data,validation_data = sampler(fold)
                train_dataset = self.dataset(train_data,**params["train_dataset"])
                val_dataset = self.dataset(validation_data,**params["val_dataset"])
                batch_size = params["loader"]["batch_size"]
                workers = params["loader"]["workers"]
                train_dataloader = data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers = workers)
                val_dataloader = data.DataLoader(val_dataset,np.int(1.5*batch_size),num_workers = workers)
                loss_fn = params["objectives"]["loss_fn"]
                score_fn = params["objectives"]["score_fn"]
                trainer = Trainer(self.device,optimizer,scheduler,train_dataloader,val_dataloader,loss_fn,score_fn)
                for epoch in range(max_epochs) :
                    if issubclass(self.scheduler,torch.optim.lr_scheduler._LRScheduler) :
                        scheduler.step()
                    train_loss = trainer.train(net)
                    val_loss = trainer.validate(net)
                    if not issubclass(self.scheduler,torch.optim.lr_scheduler._LRScheduler) :
                        scheduler.step(val_loss)
                    fold_validation.append(val_loss)
                    fold_training.append(train_loss)
                    print("output",config_id,fold,epoch,train_loss,val_loss,flush=True)
                    if curr_best >= val_loss :
                        curr_best = val_loss
                        torch.save(net.state_dict(),self.model_file.format(config_id,fold))
                validation.append(fold_validation)
                training.append(fold_training)
                del net
                torch.cuda.empty_cache()
            training = np.array(training)
            validation = np.array(validation)
            indices = np.argmax(validation,axis=1)
            avg_validation_loss = np.mean(np.max(validation,axis=1))
            avg_training_loss = np.mean(training[range(num_folds),indices])
            config_scores[config_id] = [avg_training_loss,avg_validation_loss]
            print("config output",config_id,avg_training_loss,avg_validation_loss,flush=True)
        print("saving final scores",flush=True)
        self.save_scores(config_scores)
        return config_scores
    
    def debug(self,dataloaders,dataset_ids,debug_fn) :
        config_scores = self.get_scores()
        config_ids = config_scores.keys()
        config_weights = np.array([1.0/config_scores[cid][1] for cid in config_ids])
        config_weights = config_weights/np.sum(config_weights)
        ensemble = []
        weights = []
        for i_config,cid in enumerate(config_ids) :
            params = self.get_params(cid)
            training_split = params["data"]["training_split"]
            validation_split = 1 - training_split
            num_folds = np.int(np.ceil(1.0/validation_split))
            ensemble = ensemble + [self.network(params["network"],self.model_file(cid,fold)) for fold in range(num_folds)]
            weights = weights + [config_weights[i_config]/num_folds for fold in range(num_folds)]
        weights = np.array(weights)
        weights = weights/np.sum(weights)
        debugger = Debugger(self.device,ensemble,weights,self.debug_dir,debug_fn)
        for i_data,dataloader in enumerate(dataloaders) :
            debugger(dataloader,dataset_ids[i_data])
            
            
class Trainer :
    def __init__(self,device,optimizer,scheduler,train_dataloader,val_dataloader,loss_fn,score_fn) :
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.score_fn = score_fn
        
    def train(self,net) :
        net.train()
        loss_value = 0
        for i_batch,sample_batch in enumerate(self.train_dataloader) :
            inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
            ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
            self.optimizer.zero_grad()
            outputs = net(inputs)
            loss = self.loss_fn(outputs,ground_truths)
            loss.backward()
            self.optimizer.step()
            loss_value += loss.detach().item()
        return loss_value/(i_batch+1)
            
    def validate(self,net) :
        with torch.no_grad():
            net.eval()
            score = 0
            for i_batch,sample_batch in enumerate(self.val_dataloader) :
                inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                outputs = net(inputs)
                score += self.score_fn(outputs,ground_truths).item()
            score = score/(i_batch+1)
            return score
    
    
class Debugger :
    def __init__(self,device,ensemble,weights,debug_dir,debug_fn) :
        self.ensemble = ensemble
        self.weights = weights
        self.debug_fn = debug_fn
        self.device = device
        self.debug_dir = debug_dir
        
    def debug(self,dataloader,data_id) :
        def collate(outputs) :
            num_outputs = len(outputs[0])
            final_outputs = []
            for i in range(num_outputs) :
                output = 0
                for out in outputs :
                    output = output + out[i]
                final_outputs.append(output)
            return final_outputs
        
        with torch.no_grad():
            for i_batch,sample_batch in enumerate(dataloader) :
                inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                debug_info = sample_batch['debug_info']
                outputs = []
                for i_net,net in enumerate(self.ensemble) :
                    net = net.to(self.device)
                    net.eval()
                    net_outputs = net(inputs,debug=True)
                    net_outputs = [self.weights[i_net]*output for output in net_outputs]
                    outputs.append(net_outputs)
                    net = net.cpu()
                    torch.cuda.empty_cache()
                outputs = collate(net_outputs)
                self.debug_fn(self.debug_dir,data_id,debug_info,outputs,ground_truths)