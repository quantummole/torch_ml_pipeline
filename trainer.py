# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import numpy as np
import pickle
import torch
from torch.utils import data
from tqdm import tqdm, trange

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
        self.search_strategy = config_params["search_strategy"](self.params_space,self.config_file)
        
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
    
    def cross_validate(self,dataset,max_epochs,num_configs=None) :
        config_scores = {}
        if not num_configs :
            num_configs = self.search_strategy.max_configs
        print("# configurations to search : {}".format(num_configs),flush=True)
        with trange(num_configs,desc="Configs") as config_iter :
            for config_batch in config_iter :
                config_id,params = self.search_strategy()
                training_split = params["data"]["training_split"]
                sampler = self.fold_strategy(dataset,training_split,**params.get("fold_options",{}))
                num_folds = sampler.num_folds
                validation = []
                training = []
                with trange(num_folds,desc = "Folds",leave=False) as fold_iter :
                    for fold in fold_iter :
                        fold_validation = []
                        fold_training = []
                        net = self.network(params["network"]).to(self.device)
                        optimizer = self.optimizer(net.parameters(),**params["optimizer"])
                        scheduler = self.scheduler(optimizer,**params["scheduler"])
                        curr_best = params["constants"]["val_best"]
                        train_data,validation_data = sampler(fold)
                        loss_fn = params["objectives"]["loss_fn"]
                        score_fn = params["objectives"]["score_fn"]
                        trainers = []
                        batch_size = params["loader"]["batch_size"]
                        workers = params["loader"]["workers"]
                        val_dataset = self.dataset(validation_data,mode=0,**params["data"].get("val_dataset",{}))
                        val_dataloader = data.DataLoader(val_dataset,np.int(1.5*batch_size),num_workers = workers)
        
        
                        for i in range(len(loss_fn)) :
                            train_dataset = self.dataset(train_data,mode=i+1,**params["data"].get("train_dataset",[{}]*(i+1))[i])
                            train_dataloader = data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers = workers)
                            trainer = Trainer(self.device,optimizer,scheduler,train_dataloader,val_dataloader,loss_fn[i],score_fn)
                            trainers.append(trainer)
                        with trange(max_epochs,desc="Epochs",leave=False) as epoch_iters :
                            for epoch in epoch_iters :
                                if issubclass(self.scheduler,torch.optim.lr_scheduler._LRScheduler) :
                                    scheduler.step()
                                train_loss = []
                                for mode,trainer in enumerate(trainers) :
                                    train_loss.append(trainer.train(net,mode+1))
                                val_loss = trainer.validate(net)
                                if not issubclass(self.scheduler,torch.optim.lr_scheduler._LRScheduler) :
                                    scheduler.step(val_loss)
                                fold_validation.append(val_loss)
                                fold_training.append(train_loss)
                                if curr_best >= val_loss :
                                    curr_best = val_loss
                                    torch.save(net.state_dict(),self.model_file.format(config_id,fold))
                                epoch_iters.set_postfix(best_validation_loss = curr_best, training_loss = train_loss )
                        validation.append(fold_validation)
                        training.append(fold_training)
                        del net
                        torch.cuda.empty_cache()
                        fold_iter.set_postfix(avg_validation_loss = np.mean(np.min(validation,axis=1)))
                training = np.array(training)
                validation = np.array(validation)
                avg_validation_loss = np.mean(np.min(validation,axis=1))
                val_weights = 1.0/np.min(validation,axis=1)
                val_weights = val_weights/np.sum(val_weights)
                indices = np.argmin(validation,axis=1)
                avg_training_loss = np.mean(training[range(num_folds),indices],axis=0)
                self.search_strategy.tune(avg_validation_loss)
                config_scores[config_id] = {"stats" :[avg_training_loss,avg_validation_loss],
                                             "fold_weights" : val_weights
                                            }
                
                self.save_scores(config_scores)
                config_iter.set_postfix(validation_loss = avg_validation_loss, training_loss = avg_training_loss )    
        print("saving final scores",flush=True)
        self.save_scores(config_scores)
        return config_scores
    
    def debug(self,datasets,dataset_ids,debug_fn) :
        config_scores = self.get_scores()
        config_ids = config_scores.keys()
        print("initializing ensemble weights across configurations",flush=True)
        config_weights = np.array([1.0/config_scores[cid]["stats"][-1] for cid in config_ids])
        config_weights = config_weights/np.sum(config_weights)
        ensemble = []
        weights = []
        for i_config,cid in enumerate(config_ids) :
            params = self.get_params(cid)
            batch_size = params["loader"]["batch_size"]
            workers = params["loader"]["workers"]
            num_folds = len(config_scores[cid]["fold_weights"])
            ensemble = ensemble + [self.network(params["network"],self.model_file.format(cid,fold)) for fold in range(num_folds)]
            weights = weights + [config_weights[i_config]*config_scores[cid]["fold_weights"][fold] for fold in range(num_folds)]
        weights = np.array(weights)
        weights = weights/np.sum(weights)
        print("Initiializing Debug Module",flush=True)
        debugger = Debugger(self.device,ensemble,weights,self.debug_dir,debug_fn)
        with tqdm(datasets,desc="Datasets") as datasets_ :
            for i_data, dataset in enumerate(datasets_):
                datasets_.set_postfix(dataset_id=dataset_ids[i_data])
                debug_dataset = self.dataset(dataset,mode=-1,**params["data"].get("val_dataset",{}))
                dataloader = data.DataLoader(debug_dataset,batch_size=batch_size,num_workers = workers)
                debugger.debug(dataloader,dataset_ids[i_data])
            
            
class Trainer :
    def __init__(self,device,optimizer,scheduler,train_dataloader,val_dataloader,loss_fn,score_fn) :
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.score_fn = score_fn
    def train(self,net,mode) :
        net.train()
        loss_value = 0
        with tqdm(self.train_dataloader,desc = "Training Epoch",leave=False) as loader :
            for i_batch,sample_batch in enumerate(loader) :
                inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                self.optimizer.zero_grad()
                outputs = net(inputs,mode)
                loss = self.loss_fn(outputs,ground_truths)
                loss.backward()
                self.optimizer.step()
                loss_value += loss.detach().item()
                loader.set_postfix(loss=(loss_value/(i_batch+1)), mode=mode)
        return loss_value/(i_batch+1)
            
    def validate(self,net) :
        with torch.no_grad():
            net.eval()
            score = 0
            with tqdm(self.val_dataloader,desc="Evaluation Epoch",leave=False) as loader :
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    outputs = net(inputs,mode=0)
                    score += self.score_fn(outputs,ground_truths).item()
                    loader.set_postfix(score=(score/(i_batch+1)))
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
            with tqdm(dataloader,desc="Batches",leave=False) as loader :
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    debug_info = sample_batch['debug_info']
                    outputs = []
                    for i_net,net in enumerate(tqdm(self.ensemble,desc="Ensemble",leave=False)) :
                        net = net.to(self.device)
                        net.eval()
                        net_outputs = net(inputs,mode=-1)
                        net_outputs = [self.weights[i_net]*output for output in net_outputs]
                        outputs.append(net_outputs)
                        net = net.cpu()
                        torch.cuda.empty_cache()
                    outputs = collate(outputs)
                    self.debug_fn(self.debug_dir,data_id,debug_info,outputs,ground_truths)