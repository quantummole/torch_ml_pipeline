# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:01:16 2018

@author: quantummole
"""
from trainer import Trainer
from fold_strategy import StratifiedDeterministicFold
from bootstrap_strategy import OverSample,UnderSample
from search_strategy import GridSearch
from datasets import DatasetGenerator,ImageClassificationDataset
from models import create_net, CustomNetClassification
from model_blocks import DoubleConvLayer, InceptionLayer
from loss import SupervisedMetricList, Accuracy, MarginLoss
from evaluator import Evaluator
from pipeline_components import Pipeline, PipelineOp
from config_sets import DictConfig, ExclusiveConfigs, NamedConfig, CombinerConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import numpy as np

def debug_fn(inference_file,data_id,outputs,ground_truths,debug_info) :
    image_ids = debug_info[0]
    output_vals =  np.argmax(nn.functional.softmax(outputs[0],dim=1).cpu().numpy(),axis=1)
    if len(ground_truths) == 0 :
        data = pd.DataFrame(np.hstack([np.array(image_ids).reshape(-1,1),output_vals.reshape(-1,1)]),columns=["ImageId","Label"])
    else :
        gt = ground_truths[0].cpu().numpy().reshape(-1,1)
        data = pd.DataFrame(np.hstack([np.array(image_ids).reshape(-1,1),gt,output_vals.reshape(-1,1)]),columns=["ImageId","Ground Truth","Label"])     
    data.to_csv(inference_file+"_{}.csv".format(data_id),header=False,index=False,mode="a+")

if __name__ == "__main__" :
    lr = NamedConfig(('lr', 1e-3))
    weight_decay = NamedConfig(('weight_decay', 0))
    optimizer_classes = NamedConfig(("optimizer",ExclusiveConfigs([optim.Adam,optim.Adagrad,optim.SGD])))
    optimizer_params = NamedConfig(("optimizer_params",DictConfig([lr,weight_decay])))
    optimizers = DictConfig([optimizer_classes,optimizer_params])

    cosine_annealing = NamedConfig(("scheduler",optim.lr_scheduler.CosineAnnealingLR))
    ca_T_max = NamedConfig(("T_max",40))
    eta_min = NamedConfig(("eta_min",1e-8))
    cosine_annealing_ops = NamedConfig(("scheduler_params",DictConfig([ca_T_max,eta_min])))
    cosine_annealing = DictConfig([cosine_annealing,cosine_annealing_ops])
    plateau = NamedConfig(("scheduler",optim.lr_scheduler.ReduceLROnPlateau))
    ca_T_max = NamedConfig(("patience",5))
    plateau_ops = NamedConfig(("scheduler_params",DictConfig([ca_T_max])))
    plateau = DictConfig([plateau,plateau_ops])
    schedulers = ExclusiveConfigs([cosine_annealing,plateau])
    
    evaluator_obj = Evaluator("./models/mnist_classifier",inference_fn=debug_fn)
    training_modes = NamedConfig(("modes",[1]))
    max_epochs = NamedConfig(("max_epochs",40))
    training_objectives = NamedConfig((1,SupervisedMetricList([[nn.CrossEntropyLoss()]],[[1.0]])))
    validation_objectives = NamedConfig((0,SupervisedMetricList([[Accuracy()]],[[1.0]])))
    objective_fns = NamedConfig(("objective_fns",DictConfig([training_objectives,validation_objectives])))
    constants = DictConfig([objective_fns,training_modes,max_epochs])
        
    network = NamedConfig(("network",create_net(CustomNetClassification)))

    growth_factor = NamedConfig(("growth_factor",ExclusiveConfigs([10])))
    input_dim = NamedConfig(("input_dim",28))
    final_dim = NamedConfig(("final_conv_dim",8))
    initial_channels = NamedConfig(("initial_channels",1))
    num_classes = NamedConfig(("num_classes",10))
    conv_module = NamedConfig(("conv_module",ExclusiveConfigs([DoubleConvLayer,InceptionLayer])))
    network_params = NamedConfig(("network_params",DictConfig([growth_factor,input_dim,final_dim,initial_channels,num_classes,conv_module])))

    train_loader_options = NamedConfig((1,{"shuffle":True,"batch_size":1000,"num_workers":4}))
    val_loader_options = NamedConfig((0,{"batch_size":1000,"num_workers":4}))
    loader_options = NamedConfig(("loader_options",DictConfig(
            [train_loader_options,val_loader_options])))
    network_loader_set1 = DictConfig([network_params,loader_options])



    growth_factor = NamedConfig(("growth_factor",ExclusiveConfigs([15])))
    input_dim = NamedConfig(("input_dim",28))
    final_dim = NamedConfig(("final_conv_dim",8))
    initial_channels = NamedConfig(("initial_channels",1))
    num_classes = NamedConfig(("num_classes",10))
    conv_module = NamedConfig(("conv_module",ExclusiveConfigs([DoubleConvLayer,InceptionLayer])))
    network_params = NamedConfig(("network_params",DictConfig([growth_factor,input_dim,final_dim,initial_channels,num_classes,conv_module])))

    train_loader_options = NamedConfig((1,{"shuffle":True,"batch_size":500,"num_workers":4}))
    val_loader_options = NamedConfig((0,{"batch_size":500,"num_workers":4}))
    loader_options = NamedConfig(("loader_options",DictConfig(
            [train_loader_options,val_loader_options])))
    network_loader_set2 = DictConfig([network_params,loader_options])


    growth_factor = NamedConfig(("growth_factor",ExclusiveConfigs([15])))
    input_dim = NamedConfig(("input_dim",28))
    final_dim = NamedConfig(("final_conv_dim",8))
    initial_channels = NamedConfig(("initial_channels",1))
    num_classes = NamedConfig(("num_classes",10))
    conv_module = NamedConfig(("conv_module",ExclusiveConfigs([DoubleConvLayer,InceptionLayer])))
    network_params = NamedConfig(("network_params",DictConfig([growth_factor,input_dim,final_dim,initial_channels,num_classes,conv_module])))

    train_loader_options = NamedConfig((1,{"shuffle":True,"batch_size":250,"num_workers":4}))
    val_loader_options = NamedConfig((0,{"batch_size":250,"num_workers":4}))
    loader_options = NamedConfig(("loader_options",DictConfig(
            [train_loader_options,val_loader_options])))
    network_loader_set3 = DictConfig([network_params,loader_options])
    
    
    network_loader_params = ExclusiveConfigs([network_loader_set1,network_loader_set2,network_loader_set3])
    network = DictConfig([network])    
    trainer_params = CombinerConfig([network,network_loader_params,optimizers,schedulers,constants])
    
    fold_params  = ExclusiveConfigs([{"training_split":0.8,"group_keys":["label"]}])
    
    train_transform = transforms.Compose([
        transforms.RandomAffine(10,(0.2,0.2),shear=1)
        ])
    train_transform_options = NamedConfig((1,
                                           DictConfig([
                                                   NamedConfig(("transform_sequence",
                                                                ExclusiveConfigs([train_transform,None])))])))
    validation_transform_options = NamedConfig((0,
                                                DictConfig(
                                                        [NamedConfig(
                                                                ("transform_sequence",None))])))
    execution_modes = NamedConfig(("execution_modes",[0,1,-1]))
    datasets = NamedConfig(("dataset_class_params",
                                DictConfig(
                                    [train_transform_options,validation_transform_options])))


    dataset_generator = DictConfig([datasets,execution_modes,NamedConfig(("dataset_class",ImageClassificationDataset))])

    trainer_op = PipelineOp("trainer",Trainer,trainer_params,evaluator_obj)
    datagen_op  = PipelineOp("dataloader_gen",DatasetGenerator,dataset_generator)
    input_block = PipelineOp("folds_generator",StratifiedDeterministicFold,fold_params)
    
    trainer_block = Pipeline([trainer_op],["datasets"],np.mean,None)
    datagen_block = Pipeline([datagen_op],["train_dataset","val_dataset","dataset_list"],np.mean,trainer_block)
    input_block = Pipeline([input_block],["dataset"],np.mean,datagen_block)

    PATH = "../datasets/mnist/testSet"
    import os
    image_ids = [x.replace("img_","").replace(".jpg","") for x in os.listdir(PATH)]
    image_files = [PATH+"/img_"+i+".jpg" for i in image_ids]
    test_dataset = np.hstack([np.array(image_ids).reshape(-1,1),np.array(image_files).reshape(-1,1)])
    test_dataset = pd.DataFrame(test_dataset,columns=["id","path"])
    
    print("creating dataset",flush=True)
    dataset = pd.read_csv("../datasets/mnist/train.csv",names=["path","label"])
    dataset["id"] = dataset["path"]
    print("initializing validation scheme",flush=True)
    inputs = {}

    inputs["dataset"] = dataset
    inputs['dataset_list'] = [test_dataset]
    input_block.execute(inputs,{}) 

#    scheme.infer([test_dataset,dataset],["test_scores","train_scores"],debug_fn)