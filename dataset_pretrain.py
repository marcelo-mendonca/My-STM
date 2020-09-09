#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:13:38 2020

@author: marcelo
"""

import os
import os.path
import numpy as np
from PIL import Image

import torch
#import torchvision
from torch.utils import data

import glob
import random
import json
from easydict import EasyDict as edict
from torchsample.transforms import RandomAffine
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import modal


################### Pascal VOC #####################
class VOC_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train', year='2012'):
        #../rvos-master/databases/VOC
        data_folder = 'VOC'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        self.root = root
        self.year = year        
        base_dir = os.path.join('VOCdevkit', 'VOC2012')
        voc_root = os.path.join(self.root, base_dir)        
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationObject')            
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, imset.rstrip('\n') + '.txt')
        
        with open(os.path.join(split_f), "r") as f:
            self.img_list = [x.strip() for x in f.readlines()]

    
    
    def __getitem__(self, idx):
        
        img_file = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_file = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_file).convert('RGB'))
        mask = np.array(Image.open(mask_file).convert('P'))     
        
        
        return image, mask, self.img_list[idx]


    def __len__(self):
        return len(self.img_list)



if __name__ == "__main__":
    
    print('inicio')
    trainset = VOC_dataset(data_root='../rvos-master/databases', year='2012', imset='train')
    print('trainset instanciado, lenght: ', len(trainset))
    
    trainloader = data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=1)
    print('trainloader instanciado, lenght: ', len(trainloader))
    
    dataiter = iter(trainloader)
    print('dataiter instanciado')
    
    for aa in range(5):
        image, mask, img = dataiter.next()
        print('img: ', img[0])
        #print('image: {}, label: {}, img_list: {}'.format(image,mask, img))        
        
        ff = plt.figure()
        ff.add_subplot(1,2,1)
        plt.imshow(image[0])
        ff.add_subplot(1,2,2)
        plt.imshow(mask[0])
        plt.show(block=True) 
        input("Press Enter to continue...")











