# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 22:01:16 2018

@author: quantummole
"""
from trainer import CrossValidation,DeterministicFold
from datasets import ImageSiameseDataset
from models import create_net, CustomNet2
from model_blocks import SiameseLossList

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

class siamese_loss :
    def __init__(self) :
        self.loss = nn.TripletMarginLoss()
    def __call__(self,positives,anchors,negatives) :
        loss = 0
        for negative in negatives :
            loss += self.loss(positives,anchors,negative)
        return loss
if __name__ == "__main__" :
    config_params = {"network" : create_net(CustomNet2),
                     "optimizer" : optim.Adam,
                     "scheduler" : optim.lr_scheduler.CosineAnnealingLR,
                     "model_dir" : "./models/mnist_siamese",
                     "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     "dataset" :ImageSiameseDataset,
                     "fold_strategy" : DeterministicFold
                     }
    
    params_space = {"network" : {"growth_factor" : [5,10,16],
                                 "input_dim" : 28,
                                 "initial_channels" : 1,
                                 "embedding_size" : 2
                                 },
                    "loader" : {"batch_size" : 50,
                                "workers" : 1
                                },
                    "optimizer" : {"lr" : [1e-2,1e-4,1e-5,1e-6,1e-3],
                                   "weight_decay" : 5e-2
                                   },
                    "scheduler" : {"eta_min" : 1e-8,
                                   "T_max" : 5
                                   },
                    "constants" : {"val_best" : 10},
                    "data" : {"training_split" : 0.8},
                    "train_dataset" : {"transform_sequence" : None,
                                 "classes_per_sample" : 10
                                 },
                    "val_dataset" : {"transform_sequence" : None,
                                 "classes_per_sample" : 10
                                 },
                    "objectives" : {"loss_fn" : SiameseLossList([siamese_loss],[1.0]),
                                    "score_fn" : SiameseLossList([siamese_loss],[1.0])
                                    }
                    }

    print("creating dataset",flush=True)
    dataset = pd.read_csv("../datasets/mnist/train.csv",names=["path","label"])
    print("initializing validation scheme",flush=True)
    scheme = CrossValidation(config_params,params_space)
    print("begin tuning",flush=True)
    config_scores  = scheme.cross_validate(dataset,25,5,scheme.RandomSearch)
    print("tuning completed" ,config_scores,flush=True)
    