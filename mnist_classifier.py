# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:01:16 2018

@author: quantummole
"""

from fold_strategy import ShuffleFoldGroups
from search_strategy import GridSearch
from trainer import CrossValidation
from datasets import ImageClassificationDataset
from models import create_net, CustomNet1
from loss import ClassificationLossList

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

if __name__ == "__main__" :
    config_params = {"network" : create_net(CustomNet1),
                     "optimizer" : optim.Adam,
                     "scheduler" : optim.lr_scheduler.CosineAnnealingLR,
                     "model_dir" : "./models/mnist_classifier",
                     "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "dataset" :ImageClassificationDataset,
                     "fold_strategy" : ShuffleFoldGroups,
                     "search_strategy" : GridSearch
                     }
    
    params_space = {"network" : {"growth_factor" : [5,10,16],
                                 "input_dim" : 28,
                                 "initial_channels" : 1,
                                 "num_classes" : 10
                                 },
                    "loader" : {"batch_size" : 2000,
                                "workers" : 1
                                },
                    "optimizer" : {"lr" : [1e-4,1e-3,1e-5],
                                   "weight_decay" : 5e-2
                                   },
                    "scheduler" : {"eta_min" : 1e-8,
                                   "T_max" : 5
                                   },
                    "constants" : {"val_best" : 10},
                    "data" : {"training_split" : 0.5,
                              "train_dataset" : [[{"transform_sequence" : None}]],
                              "val_dataset" : [[{"transform_sequence" : None}]],
                              },
                    "objectives" : {"loss_fn" : [[ClassificationLossList([[nn.CrossEntropyLoss]],[[1.0]])]],
                                    "score_fn" : ClassificationLossList([[nn.CrossEntropyLoss]],[[1.0]])
                                    },
                    "fold_options" : {"group_keys" : [["label"]]}
                    }

    print("creating dataset",flush=True)
    dataset = pd.read_csv("../datasets/mnist/train.csv",names=["path","label"])
    print("initializing validation scheme",flush=True)
    scheme = CrossValidation(config_params,params_space)
    print("begin tuning",flush=True)
    config_scores  = scheme.cross_validate(dataset,15)
    print("tuning completed" ,config_scores,flush=True)
    