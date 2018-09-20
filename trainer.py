# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import torch
from tqdm import tqdm, trange

class Trainer :
    def __init__(self,network,device,optimizer,scheduler,train_dataloaders,val_dataloader,modes,loss_fns,score_fn,model_file) :
        self.network = network
        self.device = device
        self.train_dataloaders = train_dataloaders
        self.val_dataloader = val_dataloader
        self.optimizer_class = optimizer
        self.scheduler_class = scheduler
        self.modes = modes
        self.loss_fns = loss_fns
        self.score_fn = score_fn
        self.model_file = model_file

    def train(self,mode) :
        self.net.train()
        loss_value = 0
        with tqdm(self.train_dataloaders[mode-1],desc = "Training Epoch",leave=False) as loader :
            for i_batch,sample_batch in enumerate(loader) :
                inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                self.optimizer.zero_grad()
                outputs = self.net(inputs,mode)
                loss = self.loss_fns[mode-1](outputs,ground_truths)
                loss.backward()
                self.optimizer.step()
                loss_value += loss.detach().item()
                loader.set_postfix(loss=(loss_value/(i_batch+1)), mode=mode)
        return loss_value/(i_batch+1)
            
    def validate(self) :
        with torch.no_grad():
            self.net.eval()
            score = 0
            with tqdm(self.val_dataloader,desc="Evaluation Epoch",leave=False) as loader :
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    outputs = self.net(inputs,mode=0)
                    score += self.score_fn(outputs,ground_truths).item()
                    loader.set_postfix(score=(score/(i_batch+1)))
                score = score/(i_batch+1)
            return score

    def fit(self,net_params,optimizer_params,scheduler_params,max_epochs,best_val_loss) :
        epoch_validations = []
        epoch_trainings = []
        self.net = self.network(net_params).to(self.device)
        self.optimizer = self.optimizer_class(self.net.parameters(),**optimizer_params)
        self.scheduler = self.scheduler_class(self.optimizer,**scheduler_params)

        with trange(max_epochs,desc="Epochs",leave=False) as epoch_iters :
            for epoch in epoch_iters :
                if issubclass(self.scheduler_class,torch.optim.lr_scheduler._LRScheduler) :
                    self.scheduler.step()
                train_loss = []
                for mode in self.modes :
                    train_loss.append(self.train(mode))
                val_loss = self.validate()
                if not issubclass(self.scheduler_class,torch.optim.lr_scheduler._LRScheduler) :
                    self.scheduler.step(val_loss)
                epoch_validations.append(val_loss)
                epoch_trainings.append(train_loss)
                if best_val_loss >= val_loss :
                    best_val_loss = val_loss
                    torch.save(self.net.state_dict(),self.model_file)
                epoch_iters.set_postfix(best_validation_loss = best_val_loss, training_loss = train_loss )
        del self.net
        torch.cuda.empty_cache()
        return epoch_trainings,epoch_validations
        
    
class Debugger :
    def __init__(self,network,device,ensemble_configs,weights,inference_dir,inference_fn,debug_dir,debug_fn) :
        self.ensemble = [network(**config) for config in ensemble_configs]
        self.weights = weights
        self.inference_fn = inference_fn
        self.debug_fn = debug_fn
        self.device = device
        self.debug_dir = debug_dir
        self.inference_dir = inference_dir
        
        
    def infer(self,dataloader,data_id) :
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
                    self.inference_fn(self.inference_dir,data_id,debug_info,outputs,ground_truths)

    def debug(self,dataloader,data_id,model_id) :
        with torch.no_grad():
            with tqdm(dataloader,desc="Batches",leave=False) as loader :
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    debug_info = sample_batch['debug_info']
                    net = self.ensemble[model_id].to(self.device)
                    net.eval()
                    net_outputs = net(inputs,mode=-2)
                    net = net.cpu()
                    torch.cuda.empty_cache()
                    self.debug_fn(self.debug_dir,data_id,debug_info,net_outputs,ground_truths)