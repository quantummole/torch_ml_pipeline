# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:01:16 2018

@author: quantummole
"""
from trainer import Trainer, Debugger
from fold_strategy import StratifiedDeterministicFold
from bootstrap_strategy import OverSample
from search_strategy import GridSearch
from validation import CrossValidationPipeline
from datasets import ImageClassificationDataset
from models import create_net, CustomNetClassification
from model_blocks import DoubleConvLayer, InceptionLayer
from loss import SupervisedMetricList, Accuracy, MarginLoss
from evaluator import Evaluator

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd
import numpy as np

def debug_fn(debug_dir,data_id,debug_info,outputs,ground_truths) :
    image_ids = debug_info[0]
    output_vals =  np.argmax(nn.functional.softmax(outputs[0],dim=1).cpu().numpy(),axis=1)
    if len(ground_truths) == 0 :
        data = pd.DataFrame(np.hstack([np.array(image_ids).reshape(-1,1),output_vals.reshape(-1,1)]),columns=["ImageId","Label"])
    else :
        gt = ground_truths[0].cpu().numpy().reshape(-1,1)
        data = pd.DataFrame(np.hstack([np.array(image_ids).reshape(-1,1),gt,output_vals.reshape(-1,1)]),columns=["ImageId","Ground Truth","Label"])
        
    data.to_csv(debug_dir+"/"+data_id+".csv",header=False,index=False,mode="a+")

if __name__ == "__main__" :
    config_params = {"network" : create_net(CustomNetClassification),
                     "optimizer" : optim.Adam,
                     "scheduler" : optim.lr_scheduler.CosineAnnealingLR,
                     "model_dir" : "./models/mnist_classifier",
                     "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "dataset" :ImageClassificationDataset,
                     "fold_strategy" : StratifiedDeterministicFold,
                     "search_strategy" : GridSearch,
                     "bootstrap_strategy" : OverSample,
                     "trainer" : Trainer,
                     "debugger" : Debugger,
                     "evaluator" : Evaluator
                     }
    train_transform = transforms.Compose([
            transforms.RandomAffine(10,(0.2,0.2),shear=1)
            ])
    params_space = {"network" : {"growth_factor" : [10,15,20],
                                 "input_dim" : 28,
                                 "final_conv_dim" : 8,
                                 "initial_channels" : 1,
                                 "num_classes" : 10,
                                 "conv_module" : [InceptionLayer,DoubleConvLayer]
                                 },
                    "loader" : {"batch_size" : 1000,
                                "workers" : 4
                                },
                    "optimizer" : {"lr" : 1e-3,
                                   "weight_decay" : 5e-2
                                   },
                    "scheduler" : {"eta_min" : 1e-8,
                                   "T_max" : 30
                                   },
                    "constants" : {"val_best" : 10},
                    "data" : {"training_split" : 0.5,
                              "train_dataset" : [{1 : { "transform_sequence" : train_transform}},{1 :{"transform_sequence" : None}}],
                              "val_dataset" : {"transform_sequence" : None},
                              },
                    "objectives" : {"loss_fn" : [([1],[SupervisedMetricList([[nn.CrossEntropyLoss(),MarginLoss(10)]],[[0.0,1.0]])]),([1],[SupervisedMetricList([[nn.CrossEntropyLoss(),MarginLoss(10)]],[[1.0,0.0]])]),([1],[SupervisedMetricList([[nn.CrossEntropyLoss(),MarginLoss(10)]],[[0.5,0.5]])])],
                                    "score_fn" : SupervisedMetricList([[Accuracy()]],[[1.0]])
                                    },
                    "fold_options" : {"group_keys" : [["label"]]},
                    "bootstrap_options" : {"group_keys" : [["label"]],
                                           "num_samples" : 2,
                                           "fraction" : 1}
                    }

    print("creating dataset",flush=True)
    dataset = pd.read_csv("../datasets/mnist/train.csv",names=["path","label"])
    dataset["id"] = dataset["path"]
    print("initializing validation scheme",flush=True)
    scheme = CrossValidationPipeline(config_params,params_space)
    print("begin tuning",flush=True)
    config_scores  = scheme.run(dataset,120)
    print("tuning completed" ,config_scores,flush=True)
    PATH = "../datasets/mnist/testSet"
    import os
    image_ids = [x.replace("img_","").replace(".jpg","") for x in os.listdir(PATH)]
    image_files = [PATH+"/img_"+i+".jpg" for i in image_ids]
    test_dataset = np.hstack([np.array(image_ids).reshape(-1,1),np.array(image_files).reshape(-1,1)])
    test_dataset = pd.DataFrame(test_dataset,columns=["id","path"])
    scheme.infer([test_dataset,dataset],["test_scores","train_scores"],debug_fn)