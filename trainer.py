# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""


import torch
from torch.autograd import Variable
from torch.utils.data import dataloader
import numpy as np
from tqdm import tqdm, trange
from signals import Signal
from functools import reduce

class Trainer :
    def __init__(self,network,network_params,
                 optimizer,optimizer_params,
                 scheduler,scheduler_params,
                 modes,loader_options,
                 evaluator,
                 max_epochs,
                 objective_fns,
                 val_max_score=1e+5,
                 adversarial_steps = 1,
                 num_batches_per_step = 1,
                 inference_iters = None,
                 patience = None) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = network
        self.network_params = network_params
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.scheduler_class = scheduler
        self.modes = modes
        self.loader_options = loader_options
        self.Evaluator = evaluator
        self.objective_fns = objective_fns
        for key in self.objective_fns.keys() :
            self.objective_fns[key] = self.objective_fns[key].to(self.device)
        self.model_file = self.Evaluator.get_model_file()
        self.net = self.network(**self.network_params).to(self.device)
        objective_fn_params = [list(fn.parameters()) for key,fn in self.objective_fns.items()]
        objective_fn_params = reduce(lambda x,y : x+y,objective_fn_params)
        self.optimizer = self.optimizer_class(list(self.net.parameters())+objective_fn_params,**self.optimizer_params)
        self.scheduler = self.scheduler_class(self.optimizer,**self.scheduler_params)
        self.max_epochs = max_epochs
        self.val_max_score = val_max_score
        self.adversarial_steps = adversarial_steps
        self.num_batches_per_step = num_batches_per_step
        self.patience = patience if patience else int(0.2*self.max_epochs)
        self.patience_counter = 0
        self.inference_iters = inference_iters
    def train(self,mode) :
        self.net.train()
        self.objective_fns[mode].train()
        loss_value = 0
        def train_closure(inputs,ground_truths) :
            loss_val = 0.0
            inputs = [Variable(inp.to(self.device), requires_grad=True) for inp in inputs]
            ground_truths = [gt.to(self.device) for gt in ground_truths]
            for j in range(self.adversarial_steps+1) :
                outputs = self.net(inputs=inputs,mode=mode)
                loss = self.objective_fns[mode](outputs,ground_truths)/(self.adversarial_steps+1)/self.num_batches_per_step
                loss.backward()
                loss_val += loss
                grads = [torch.ge(inp.grad,0.0).type_as(inp) for inp in inputs]
                inputs = [Variable(inp.data + 0.007*(grad-0.5),requires_grad=True) for inp,grad in zip(inputs,grads)]
            return loss_val
        def step_closure(input_batches,gt_batches) :
            loss_val = 0.0 
            for inp,gt in zip(input_batches,gt_batches) :
                loss_val += train_closure(inp,gt)
            return  loss_val
        input_batches = []
        gt_batches = []
        self.optimizer.zero_grad()
        step_counter = 0
        with tqdm(self.dataloaders[mode],desc = "Training Epoch") as loader :
            for i_batch,sample_batch in enumerate(loader) :
                input_batches.append(sample_batch['inputs'])
                gt_batches.append(sample_batch['ground_truths'])
                step_counter += 1
                if step_counter == self.num_batches_per_step :
                    closure = lambda : step_closure(input_batches,gt_batches)
                    self.optimizer.step(closure)
                    loss_value += closure().detach().item()
                    input_batches = []
                    gt_batches = []
                    self.optimizer.zero_grad()
                    step_counter = 0
                loader.set_postfix(loss=(loss_value/(i_batch+1)), mode=mode)
        if step_counter > 0 :
            closure = lambda : step_closure(input_batches,gt_batches)
            self.optimizer.step(closure)
            loss_value += np.sum([train_closure(inp,gt) for inp,gt in zip(input_batches,gt_batches)])
            input_batches = []
            gt_batches = []
            self.optimizer.step(closure)
            loss_value += closure().detach().item()
            input_batches = []
            gt_batches = []
            self.optimizer.zero_grad()
            step_counter = 0            
        return loss_value/(i_batch+1)*self.num_batches_per_step
            
    def validate(self) :
        with torch.no_grad():
            self.net.eval()
            self.objective_fns[0].eval()
            score = 0
            with tqdm(self.dataloaders[0],desc="Evaluation Epoch") as loader :
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    outputs = self.net(inputs=inputs,mode=0)
                    score += self.objective_fns[0](outputs,ground_truths).item()
                    loader.set_postfix(score=(score/(i_batch+1)))
                score = score/(i_batch+1)
            return score

    def infer(self) :
        self.net.load_state_dict(torch.load(self.model_file,map_location=lambda storage, loc: storage))
        for data_id,loader in enumerate(self.dataloaders['inference']) :
            with torch.no_grad():
                self.net.eval()
                for i_batch,sample_batch in enumerate(loader) :
                    inputs = [inp.to(self.device) for inp in sample_batch['inputs']]
                    ground_truths = [gt.to(self.device) for gt in sample_batch['ground_truths']]
                    debug_info = sample_batch['debug_info']
                    outputs = []
                    outputs = self.net(inputs=inputs,mode=0)
                    self.Evaluator.log("inference",data_id,[output.detach().cpu() for output in outputs],[gt.detach().cpu() for gt in ground_truths],debug_info)

    def execute(self,datasets) :
        epoch_scores = []
        self.dataloaders = {}
        get = lambda myD,key : myD[key] if key in myD else myD[0]
        for mode in datasets.keys() :
            dataset = datasets[mode]
            if not isinstance(dataset,list) :
                self.dataloaders[mode] = dataloader.DataLoader(dataset,**get(self.loader_options,mode))
            else :
                self.dataloaders[mode] = [dataloader.DataLoader(data,**get(self.loader_options,mode)) for data in dataset]
        best_epoch = -1
        with trange(self.max_epochs,desc="Epochs") as epoch_iters :
            for epoch in epoch_iters :
                if issubclass(self.scheduler_class,torch.optim.lr_scheduler._LRScheduler) :
                    self.scheduler.step()
                train_loss = []
                for mode in self.modes :
                    train_loss.append(self.train(mode))
                val_loss = self.validate()
                scores = [val_loss] + train_loss
                epoch_scores.append(scores)
                if not issubclass(self.scheduler_class,torch.optim.lr_scheduler._LRScheduler) :
                    self.scheduler.step(val_loss)
                if self.val_max_score >= val_loss :
                    self.val_max_score = val_loss
                    best_epoch = epoch
                    self.patience_counter = 0
                    torch.save(self.net.state_dict(),self.model_file)
                if not best_epoch == epoch :
                    self.patience_counter += 1
                    if self.patience_counter == self.patience :
                        break
                epoch_iters.set_postfix(best_epoch = best_epoch,best_validation_loss =  self.val_max_score , training_loss = train_loss, config_id = self.Evaluator.config_id)
        self.inference_iters = self.inference_iters if self.inference_iters else 0.1*best_epoch
        for i in range(self.inference_iters) :
            self.infer()
        del self.net
        torch.cuda.empty_cache()
        return Signal.COMPLETE,[epoch_scores]
        
    
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
