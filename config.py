#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:50:51 2020

@author: marcelo
"""

import os
from easydict import EasyDict as edict


__C = edict()

# Public access to configuration settings
cfg = __C

# GPUs to be used
__C.GPU = "0" #“0,2,5” to use only special devices (note, that in this case, pytorch will count all available devices as 0,1,2 )
os.environ['CUDA_VISIBLE_DEVICES'] = __C.GPU

# Root folder of datasets
__C.DATA_ROOT = os.path.abspath('../rvos-master/databases')

# Trainset image sizes
__C.TRAIN_IMG_SIZE = (384, 384)

# Valset image sizes
__C.VAL_IMG_SIZE = (384, 384)

# Directory to save checkpoints
__C.MODEL_DIR = 'saved_models'

# Number of epochs for main train
__C.MAIN_TRAIN_EPOCHS = 60

# Number of epochs for pre-train
__C.PRE_TRAIN_EPOCHS = 60

# Interval of epochs to perform validation
__C.EVAL_EPOCHS = 10

# Interval of epochs to save checkpoint
__C.CHECKPOINT_EPOCHS = 10

# Datasets for main train
__C.MAIN_TRAIN_DATASETS = {'youtube': 1, 'davis': 5}

# Datasets for validation
__C.VAL_DATASETS = {'youtube': 1, 'davis': 1}

# Datasets for pre train
__C.PRE_TRAIN_DATASETS = {'voc': 1, 'ecssd': 1, 'msra': 1, 'sbd': 1, 'coco': 1}

# learning rate
__C.lr = 1e-5















