# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""

from torch.utils.data import Dataset, dataloader
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
import torchvision.datasets as datasets


#mode = -1 is for test and debug
#mode = 0 is for validation
#mode = {1,2,3..} is for training

from signals import Signal
class DatasetGenerator :
    def __init__(self,execution_modes,dataset_class,dataset_class_params,loader_options) :
        self.dataset_class = dataset_class
        self.dataset_class_params = dataset_class_params
        self.loader_options = loader_options
        self.modes = execution_modes
    def execute(self,dataloaders=None,train_dataset=None,val_dataset=None,dataset=None) :
        if dataloaders == None :
            dataloaders = {}
        for mode in self.modes :
            if mode > 0 :
                dataset = train_dataset
            elif mode == 0 :
                dataset = val_dataset
            else :
                dataset = dataset
            dataset = self.dataset_class(data=dataset,mode=mode,**self.dataset_class_params[mode])
            dataset = dataloader.DataLoader(dataset,**self.loader_options)
            dataloaders[mode] = dataset
        return Signal.COMPLETE,'#'.join([str(i) for i in self.modes]),[dataloaders]

class ImageClassificationDataset(Dataset) :
    def __init__(self,data,mode = -1,transform_sequence = None) :
        super(ImageClassificationDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data.id.values.tolist()
        self.image_paths = data.path.values.tolist()
        if not self.mode == -1 :
            self.image_class = data.label.values.tolist()
        else :
            if "label" in data.columns:
                self.image_class = data.label.values.tolist()
            else :
                self.image_class = None
    def __len__(self) :
        return len(self.image_paths)
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        im = io.imread(path)
        img = Image.fromarray(im)
        if self.transform_sequence :
            img = self.transform_sequence(img)
        im = (np.array(img)/np.max(im)*1.0).astype(np.float32)
        gt = []
        if not self.mode == -1 :
            label = np.long(self.image_class[idx])
            gt = [label]
        else :
            gt =[]
            if self.image_class :
                gt = [np.long(self.image_class[idx])]
        return {"inputs":[im],"ground_truths":gt,"debug_info":[self.image_id[idx]]}

class ImageSiameseDataset(Dataset) :
    def __init__(self,data,classes_per_sample,mode = -1,transform_sequence = None) :
        super(ImageSiameseDataset,self).__init__()
        self.data_groups = data.groupby(["label"])
        self.groups = list(self.data_groups.groups.keys())
        self.group_counts = self.data_groups.count().values
        self.num_groups = len(self.data_groups.groups.keys())
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.transform_sequence = transform_sequence
        self.classes_per_sample = classes_per_sample
        self.mode = mode
    def __len__(self) :
        return np.min(self.group_counts)
    def __getitem__(self,idx) :
        classes =  np.random.choice(self.groups,size=min([self.classes_per_sample,self.num_groups]),replace=False)
        inputs = []
        labels = []
        for cls in classes :
            paths = np.random.choice(self.data_groups.get_group(cls)["path"].values,size=2,replace=False)
            for path in paths :
                im = io.imread(path)
                img = Image.fromarray(im)
                if self.transform_sequence :
                    img = self.transform_sequence(img)
                im = (np.array(img)/np.max(im)*1.0).astype(np.float32)
                inputs.append(im)
                labels.append(cls)
        return {"inputs":inputs,"ground_truths":labels}


def clean_null(x,y) :
    return x if pd.notnull(x) else y
 
def create_csv_file(root_dir) :
    dataset =  datasets.ImageFolder(root_dir)
    z = dataset.imgs
    x = pd.DataFrame(np.array(z),columns=["path","label"])
    x.to_csv(root_dir+"/../train.csv",index=False,header = False)