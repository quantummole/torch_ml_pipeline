# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 11:02:42 2018

@author: quantummole
"""

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import io
import pydicom
from PIL import Image
import torchvision.datasets as datasets

class ImageClassificationDataset(Dataset) :
    def __init__(self,data,mode = -1,transform_sequence = None) :
        super(ImageClassificationDataset,self).__init__()
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.mode = mode
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

class FGClassificationDataset(Dataset) :
    def __init__(self,data,mode = -1,transform_sequence = None) :
        super(FGClassificationDataset,self).__init__()
        self.image_paths = data.path.values.tolist()
        self.image_class = data.label.values.tolist()
        self.image_xs = data.x.values.tolist()
        self.image_ys = data.y.values.tolist()
        self.image_hs = data.height.values.tolist()
        self.image_ws = data.width.values.tolist()
        self.transform_sequence = transform_sequence
        self.mode = mode
    def __len__(self) :
        return len(self.image_class)
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        gts = []
        im = pydicom.read_file(path).pixel_array
        mask = np.zeros_like(im)
        label = np.long(self.image_class[idx])
        x = np.int(clean_null(self.image_xs[idx],0))
        y = np.int(clean_null(self.image_ys[idx],0))
        h = np.int(clean_null(self.image_hs[idx],0))
        w = np.int(clean_null(self.image_ws[idx],0))
        mask[y:y+h,x:x+w] = 1.0       
        m,n = im.shape 
        for i in range(0,m,32) :
           for j in range(0,n,32) :
               gts.append(np.long(1*(np.mean(mask[i:i+32,j:j+32]) > 0.1)))
        img = Image.fromarray(im)
        if self.transform_sequence :
            img = self.transform_sequence(img)
        im = (np.array(img)/np.max(im)*1.0).astype(np.float32)

        return {"inputs":[im],"ground_truths":[label,np.array(gts).reshape((m//32,n//32))]}


def clean_null(x,y) :
    return x if pd.notnull(x) else y
 
def create_csv_file(root_dir) :
    dataset =  datasets.ImageFolder(root_dir)
    z = dataset.imgs
    x = pd.DataFrame(np.array(z),columns=["path","label"])
    x.to_csv(root_dir+"/../train.csv",index=False,header = False)