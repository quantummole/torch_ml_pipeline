# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import numpy as np
import pickle
from torch.utils import data
from tqdm import tqdm, trange

class CrossValidationPipeline :
    def __init__(self,config_params,params_space) :
        self.params_space = params_space
        self.network = config_params["network"]
        self.optimizer = config_params["optimizer"]
        self.scheduler = config_params["scheduler"]
        self.search_strategy = config_params["search_strategy"]
        self.fold_strategy = config_params["fold_strategy"]
        self.bootstrap = config_params["bootstrap_strategy"]
        self.device = config_params["device"]
        self.dataset = config_params["dataset"]
        self.Trainer = config_params["trainer"]
        self.Debugger = config_params["debugger"]
        
        self.model_dir = config_params["model_dir"]+"/models"
        self.model_file = self.model_dir+"/model_{}_{}_{}.mod"
        self.debug_dir = config_params["model_dir"]+"/debug/"
        self.config_dir = config_params["model_dir"]+"/configs"
        self.config_file = self.config_dir+"/config_{}.pkl"
        self.inference_dir = config_params["model_dir"]+"/inference/"
        self.score_file = config_params["model_dir"]+"/score.pkl"

        self.get_model_file = lambda config_id : lambda sample_id : lambda fold_id : self.model_file.format(config_id,sample_id,fold_id)

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

    def run(self,dataset,max_epochs,num_configs=None) :
        config_scores = {}
        searcher = self.search_strategy(self.params_space,self.config_file)
        if not num_configs :
            num_configs = searcher.max_configs
        print("# configurations to search : {}".format(num_configs),flush=True)        
        with trange(num_configs,desc="Configs") as config_iter :
            for config_batch in config_iter :
                config_id,params = searcher()
                avg_training_loss,avg_validation_loss,batch_weights,fold_weights = self.bootstrap_validation(params,dataset,max_epochs,self.get_model_file(config_id))
                searcher.tune(avg_validation_loss)
                config_scores[config_id] = {"stats" :[avg_training_loss,avg_validation_loss],
                                             "batch_weights" : batch_weights,
                                             "fold_weights" : fold_weights
                                             }
                self.save_scores(config_scores)
                config_iter.set_postfix(validation_loss = avg_validation_loss, training_loss = avg_training_loss, val_deviation =  np.std(batch_weights))   
        print("saving final scores",flush=True)
        self.save_scores(config_scores)
        return config_scores

    def bootstrap_validation(self,params,dataset,max_epochs,model_file) :
        bootstrapper = self.bootstrap(dataset,**params.get("bootstrap_options",{}))
        num_samples = bootstrapper.num_samples
        validation = []
        training = []
        fold_weights = []
        with trange(num_samples,desc="Bootstrap Samples") as samples_iter :
            for sample_id in samples_iter :
                subset = bootstrapper.sample(sample_id)
                avg_training_loss,avg_validation_loss,val_weights = self.cross_validate(params,subset,max_epochs,model_file(sample_id))
                validation.append(avg_validation_loss)
                training.append(avg_training_loss)
                fold_weights.append(val_weights)
                samples_iter.set_postfix(avg_training_loss = avg_training_loss, avg_validation_loss = avg_validation_loss, val_deviation = np.std(val_weights))
        validation = np.array(validation)
        training = np.array(training)
        batch_weights = 1.0/validation
        batch_weights = batch_weights/np.sum(batch_weights)
        avg_validation_loss = np.mean(validation)
        avg_training_loss = np.mean(training,axis=0)
        return avg_training_loss,avg_validation_loss,batch_weights,fold_weights        

    def cross_validate(self,params,dataset,max_epochs,model_file) :
            training_split = params["data"]["training_split"]
            sampler = self.fold_strategy(dataset,training_split,**params.get("fold_options",{}))
            num_folds = sampler.num_folds
            validation_scores = []
            training_scores = []
            with trange(num_folds,desc = "Folds",leave=False) as fold_iter :
                for fold in fold_iter :
                    val_best = params["constants"]["val_best"]
                    train_data,validation_data = sampler(fold)
                    modes,loss_fns = params["objectives"]["loss_fn"]
                    score_fn = params["objectives"]["score_fn"]
                    batch_size = params["loader"]["batch_size"]
                    workers = params["loader"]["workers"]

                    val_dataset = self.dataset(validation_data,mode=0,**params["data"].get("val_dataset",{}))
                    train_datasets = [self.dataset(train_data,mode=mode,**params["data"].get("train_dataset",{mode : {}})[mode]) for mode in modes]

                    val_dataloader = data.DataLoader(val_dataset,np.int(1.5*batch_size),num_workers = workers)
                    train_dataloaders = [data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers = workers) for train_dataset in train_datasets]

                    trainer = self.Trainer(self.network,self.device,self.optimizer,self.scheduler,train_dataloaders,val_dataloader,modes,loss_fns,score_fn,model_file(fold))
                    fold_training,fold_validation = trainer.fit(params["network"],params["optimizer"],params["scheduler"],max_epochs,val_best)
                    validation_scores.append(fold_validation)
                    training_scores.append(fold_training)
                    del trainer
            training = np.array(training_scores)
            validation = np.array(validation_scores)
            avg_validation_loss = np.mean(np.min(validation,axis=1))
            val_weights = 1.0/np.min(validation,axis=1)
            val_weights = val_weights/np.sum(val_weights)
            indices = np.argmin(validation,axis=1)
            avg_training_loss = np.mean(training[range(num_folds),indices],axis=0)
            return avg_training_loss,avg_validation_loss,val_weights
    
    def infer(self,datasets,dataset_ids,inference_fn,debug_fn = None) :
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
            num_batches = len(config_scores[cid]["batch_weights"])
            ensemble = ensemble + [{**{"network_params" : params["network"]},**{"weights" : self.model_file.format(cid,sample_id,fold)}} for sample_id in range(num_batches) for fold in range(len(config_scores[cid]["fold_weights"][sample_id]))]
            weights = weights + [config_weights[i_config]*config_scores[cid]["batch_weights"][sample_id]*config_scores[cid]["fold_weights"][sample_id][fold] for sample_id in range(num_batches) for fold in range(len(config_scores[cid]["fold_weights"][sample_id]))]
        weights = np.array(weights)
        weights = weights/np.sum(weights)
        print("Initiializing Debug Module",flush=True)
        debugger = self.Debugger(self.network,self.device,ensemble,weights,self.inference_dir,inference_fn,self.debug_dir,debug_fn)
        with tqdm(datasets,desc="Datasets") as datasets_ :
            for i_data, dataset in enumerate(datasets_):
                datasets_.set_postfix(dataset_id=dataset_ids[i_data])
                debug_dataset = self.dataset(dataset,mode=-1,**params["data"].get("val_dataset",{}))
                dataloader = data.DataLoader(debug_dataset,batch_size=batch_size,num_workers = workers)
                debugger.infer(dataloader,dataset_ids[i_data])