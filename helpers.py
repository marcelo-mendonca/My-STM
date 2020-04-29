from __future__ import division
#torch
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import copy


class font:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)

def iou(pred, gt):
    #pred:  torch.Size([1, 11, 3, 100, 100])
    pred = pred.squeeze().cpu().data.numpy()
    #pred:  (11, 3, 100, 100)
    pred = ToLabel(pred)
    #pred:  (3, 100, 100)
    gt = gt.squeeze().cpu().data.numpy()
    #gt:  (11, 3, 100, 100)
    agg = pred + gt
    i = float(np.sum(agg == 2))
    print("#i: ", i.shape)
    u = float(np.sum(agg > 0))
    print("#u: ", u.shape)
    for i in range(11):
        for j in range(3):
            #Ms:  torch.Size([4, 11, 3, 480, 854])
            #pred:  (3, 100, 100)
            #agg:  (11, 3, 100, 100)
            print("#pred: ", pred.shape)
            print("#agg: ", agg.shape)
            plt.matshow(pred[i,j])
            plt.show()
            input("Press Enter to continue...")
            plt.matshow(agg[i,j])
            plt.show()
            input("Press Enter to continue...")
    
    return i / u

def ToLabel(E):
    fgs = np.argmax(E, axis=0).astype(np.float32)
    return fgs.astype(np.uint8)



