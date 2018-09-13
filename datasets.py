# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
import torchvision.datasets as datasets

class ImageClassificationDataset(Dataset) :
    def __init__(self,data,transform_sequence = None) :
        super(ImageClassificationDataset,self).__init__()
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.transform_sequence = transform_sequence
    def __len__(self) :
        return len(self.image_class)
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        im = io.imread(path)
        img = Image.fromarray(im)
        if self.transform_sequence :
            img = self.transform_sequence(img)
        im = (np.array(img)/np.max(im)*1.0).astype(np.float32)
        label = np.long(self.image_class[idx])
        return {"inputs":[im],"ground_truths":[label]}

class ImageSiameseDataset(Dataset) :
    def __init__(self,data,classes_per_sample,transform_sequence = None) :
        super(ImageSiameseDataset,self).__init__()
        self.data_groups = data.groupby(["label"])
        self.groups = list(self.data_groups.groups.keys())
        self.group_counts = self.data_groups.count().values
        self.num_groups = len(self.data_groups.groups.keys())
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.transform_sequence = transform_sequence
        self.classes_per_sample = classes_per_sample
    def __len__(self) :
        return np.min(self.group_counts)
    def __getitem__(self,idx) :
        classes =  np.random.choice(self.groups,size=self.classes_per_sample,replace=False)
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

def create_csv_file(root_dir) :
    dataset =  datasets.ImageFolder(root_dir)
    z = dataset.imgs
    x = pd.DataFrame(np.array(z),columns=["path","label"])
    x.to_csv(root_dir+"/../train.csv",index=False,header = False)