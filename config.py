#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 13:50:51 2020

@author: marcelo
"""

import os
from easydict import EasyDict as edict

# pre_train loader lenght:  139249 [VOC + ECSSD + MSRA + SBD + COCO]
# pre_val loader lenght:  10770 [VOC + SBD + COCO]
# main_train loader lenght:  91777 [5*DAVIS + YOUTUBE]
# main_val loader lenght:  27607 [5*DAVIS + YOUTUBE]

__C = edict()

# Public access to configuration settings
cfg = __C

# GPUs to be used
__C.GPU = "0" #“0,2,5” to use only special devices (note, that in this case, pytorch will count all available devices as 0,1,2 )
os.environ['CUDA_VISIBLE_DEVICES'] = __C.GPU

# Root folder of datasets
__C.DATA_ROOT = os.path.abspath('../rvos-master/databases')

# Youtube data folder
__C.DATA_YOUTUBE = os.path.join(__C.DATA_ROOT, 'YouTubeVOS')

# DAVIS data folder
__C.DATA_DAVIS = os.path.join(__C.DATA_ROOT, 'DAVIS2017')

# Pascal VOC data folder
__C.DATA_VOC = os.path.join(__C.DATA_ROOT, 'VOC', 'VOCdevkit', 'VOC2012')

# ECSSD data folder
__C.DATA_ECSSD = os.path.join(__C.DATA_ROOT, 'ECSSD')

# MSRA data folder
__C.DATA_MSRA = os.path.join(__C.DATA_ROOT, 'MSRA', 'MSRA10K_Imgs_GT')

# SBD data folder
__C.DATA_SBD = os.path.join(__C.DATA_ROOT, 'SBD', 'benchmark_RELEASE', 'dataset')

# COCO data folder
__C.DATA_COCO = os.path.join(__C.DATA_ROOT, 'COCO')

# Trainset image sizes
__C.TRAIN_IMG_SIZE = (150, 150) #(384, 384)

# Valset image sizes
__C.VAL_IMG_SIZE = (150, 150) #(384, 384)

# Directory to save checkpoints
__C.MODEL_DIR = 'saved_models'

# Number of epochs for main train
__C.MAIN_TRAIN_EPOCHS = 200

# Number of epochs for pre-train
__C.PRE_TRAIN_EPOCHS = 200

# Interval of epochs to perform validation
__C.EVAL_EPOCHS = 10

# Interval of epochs to save checkpoint
__C.CHECKPOINT_EPOCHS = 300

# Interval of iterations to log and display partials
__C.DISP_INTERVAL = 5

# Datasets for main train
__C.MAIN_TRAIN_DATASETS = {'youtube': 1, 'davis': 5}

# Datasets for validation during main train
__C.MAIN_VAL_DATASETS = {'youtube': 1, 'davis': 1}

# Datasets for pre-train
__C.PRE_TRAIN_DATASETS = {'voc': 1, 'ecssd': 1, 'msra': 1, 'sbd': 1, 'coco': 1}

# Datasets for validation during pre-train
__C.PRE_VAL_DATASETS = {'voc': 1, 'sbd': 1, 'coco': 1}

# learning rate
__C.LR = 1e-5

# Flag to enable dataset_pretrain script
__C.PRETRAIN = False















