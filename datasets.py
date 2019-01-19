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
import pydicom
#mode = -1 is for test and debug
#mode = 0 is for validation
#mode = {1,2,3..} is for training

from signals import Signal

def im_reader(path) :
    if ".dcm" in path :
       im = pydicom.read_file(path).pixel_array 
    else :
        im = io.imread(path) 
    im = im/(np.max(im)+1e-5)
    return im

class DatasetGenerator :
    def __init__(self,execution_modes,dataset_class,dataset_class_params,evaluator = None) :
        self.dataset_class = dataset_class
        self.dataset_class_params = dataset_class_params
        self.modes = execution_modes
    def execute(self,datasets=None,train_dataset=None,val_dataset=None,dataset_list=[]) :
        if datasets == None :
            datasets = {}
        for mode in self.modes :
            if mode > 0 :
                dataset = train_dataset
            elif mode == 0 :
                dataset = val_dataset
            else :
                dataset = dataset_list
            if mode >= 0 :
                dataset = self.dataset_class(data=dataset,mode=mode,**self.dataset_class_params[mode])
            else :
                dataset_list = [val_dataset]+dataset_list
                dataset = [self.dataset_class(data=dataset,mode=0,**self.dataset_class_params[0]) for dataset in dataset_list]
                mode = 'inference'
            datasets[mode] = dataset
        return Signal.COMPLETE,[datasets]

class ImageClassificationDataset(Dataset) :
    def __init__(self,data,mode = -1,reader = im_reader,label_columns = None,transform_sequence = None) :
        super(ImageClassificationDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data.id.values.tolist()
        self.image_paths = data.path.values.tolist()
        if label_columns :
            self.image_class = data[label_columns].values.tolist()
        else :
            self.image_class = None
        self.reader = reader
    def __len__(self) :
        return len(self.image_paths)
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        img = self.reader(path)
        if self.transform_sequence :
            img = self.transform_sequence(img)
        im = np.array(img)
        im = (im/np.max(im)*1.0).astype(np.float32)
        gt = []
        if self.image_class :
            label = np.long(self.image_class[idx])
            gt = [label]
        return {"inputs":[im],"ground_truths":gt,"debug_info":[self.image_id[idx]]}

class ImageSiameseDataset(Dataset) :
    def __init__(self,data,classes_per_sample,mode = -1,reader = im_reader,transform_sequence = None) :
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
        self.reader = reader
    def __len__(self) :
        return np.min(self.group_counts)
    def __getitem__(self,idx) :
        classes =  np.random.choice(self.groups,size=min([self.classes_per_sample,self.num_groups]),replace=False)
        inputs = []
        labels = []
        for cls in classes :
            paths = np.random.choice(self.data_groups.get_group(cls)["path"].values,size=2,replace=False)
            for path in paths :
                img = self.reader(path)
                if self.transform_sequence :
                    img = self.transform_sequence(img)
                im = (np.array(img)/np.max(img)*1.0).astype(np.float32)
                inputs.append(im)
                labels.append(cls)
        return {"inputs":inputs,"ground_truths":labels}

class ImageSegmentationDataset(Dataset) :
    def __init__(self,data,mode = -1,image_reader=im_reader,mask_reader=im_reader,transform_sequence = None) :
        super(ImageSegmentationDataset,self).__init__()
        self.mode = mode
        self.transform_sequence = transform_sequence
        self.image_id = data['id'].values.tolist()
        self.image_paths = data['path'].values.tolist()
        if "mask" in data.columns:
            self.image_mask = data['mask'].values.tolist()
        else :
            self.image_mask = None
        self.image_reader = image_reader
        self.mask_reader = mask_reader
    def __len__(self) :
        return len(self.image_paths)
    def __getitem__(self,idx) :
        path = self.image_paths[idx]
        img = self.image_reader(path)
        gt = []
        if self.image_mask :
            path = self.image_mask[idx]
            label = self.mask_reader(path)
            label = label.astype(np.int64)
            gt = [label]
        if self.transform_sequence :
            if self.image_mask :
                max_l = np.max(gt[0])
                gt_img = gt[0]
                img,gt_img = self.transform_sequence([img,gt_img])
                gt = np.array(gt_img)
                gt = np.clip(gt.round(),0,max_l).astype(np.int64)
                gt = [gt,gt] 
                if self.mode < 1 :
                    gt = gt[1:]
            else :
                img = self.transform_sequence(img)
        im = np.array(img)
        im = (im/np.max(im+1e-5)*1.0).astype(np.float32)

        return {"inputs":[im],"ground_truths":gt,"debug_info":[self.image_id[idx]]}

def clean_null(x,y) :
    return x if pd.notnull(x) else y

def create_csv_file(root_dir) :
    dataset =  datasets.ImageFolder(root_dir)
    z = dataset.imgs
    x = pd.DataFrame(np.array(z),columns=["path","label"])
    x.to_csv(root_dir+"/../train.csv",index=False,header = False)