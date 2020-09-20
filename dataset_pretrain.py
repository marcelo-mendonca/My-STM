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
from torchsample.transforms import RandomAffine, RandomCrop, ToTensor, ChannelsFirst, AddChannel
from torchsample.transforms import ChannelsLast, Compose, RandomBrightness, RangeNormalize, TypeCast
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import modal
from pycocotools.coco import COCO
from itertools import cycle


############################ Coco Background ################################  
class coco_background(data.Dataset):
    
    def __init__(self, data_root, imset='val', year='2017'):
        data_folder = 'COCO'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        self.root = root 
        set_year = '{}{}'.format(imset,year)
        self.image_dir = os.path.join(self.root, 'images',set_year)
        ann_dir = os.path.join(self.root, 'annotations')
        self.annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
        
        self.coco = COCO(self.annFile)
        self.img_ids = self.coco.getImgIds()
        self.fixed_size = (384, 384)
        
        #img:  [{'license': 5, 'file_name': '000000010092.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000010092.jpg', 
        #'height': 426, 'width': 640, 'date_captured': '2013-11-21 00:20:22', 
        #'flickr_url': 'http://farm9.staticflickr.com/8276/8710590452_08a7a8f59c_z.jpg', 'id': 10092}]        
        
    def __getitem__(self, idx):
        
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs(img_id)
        img_path = os.path.join(self.image_dir,img_data[0]['file_name'])
        #img = Image.open(img_path).convert('RGB')        
        #image = np.array(img)/255.
        #image = np.array(img.resize(self.fixed_size, Image.ANTIALIAS))/255.        
        w = img_data[0]['width']
        h = img_data[0]['height']       
        
        return img_path, w, h
    
    def __len__(self):
        return len(self.img_ids)

####################### Instanciate global objects########################## 
print('COCO Dataset----------------------')
coco_bg = coco_background(data_root='../rvos-master/databases')
coco_bg_loader = data.DataLoader(coco_bg, batch_size=1, shuffle=True, num_workers=1)
coco_bg_cycle = cycle(coco_bg_loader)
print('-------------------------Completed!')

bg_affine = RandomAffine(rotation_range=20, shear_range=10, zoom_range=(0.9, 1.1))
    
bg_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), RangeNormalize(0, 1), 
                     RandomBrightness(-0.2,0.2), bg_affine, RandomCrop(coco_bg.fixed_size), ChannelsLast()])

fg_affine = RandomAffine(rotation_range=90, shear_range=10, zoom_range=(0.5, 1.5))

fg_rgb_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                     RangeNormalize(0, 1), RandomBrightness(-0.2,0.2)])

fg_mask_transf = Compose([ToTensor(), AddChannel(axis=0), TypeCast('float'), 
                          RangeNormalize(0, 1)])

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
        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'))
        mask = (mask != 255).astype(np.uint8) * mask
        num_objects = np.max(mask)
        
        global coco_bg_cycle
        bg_path, _, _ = next(coco_bg_cycle)
        
        bg = Image.open(bg_path[0]).convert('RGB')
        
        make_triplet2(image, mask, num_objects, np.array(bg))
        
        
        return image, mask, num_objects, np.array(bg)


    def __len__(self):
        return len(self.img_list)
    
    

############################ AUX FUNCTIONS ################################

def make_triplet2(fg_frame, fg_mask, num_objects, bg=None, triplet_size=(384,384)): 
    
    bg_triplet = make_bg_triplet(bg, triplet_size)
    
    N_frames, N_masks, num_objects = make_fg_triplet(fg_frame, fg_mask, num_objects, bg_triplet)
    
    ff = plt.figure()
    ff.add_subplot(2,3,1)
    plt.imshow(N_frames[0])
    ff.add_subplot(2,3,2)
    plt.imshow(N_frames[1])
    ff.add_subplot(2,3,3)
    plt.imshow(N_frames[2])
    ff.add_subplot(2,3,4)
    plt.imshow(N_masks[0])
    ff.add_subplot(2,3,5)
    plt.imshow(N_masks[1])
    ff.add_subplot(2,3,6)
    plt.imshow(N_masks[2])
    plt.show(block=True) 
    input("Press Enter to continue...")   
    
    
    return        

def make_fg_triplet(fg_frame, fg_mask, num_objects, bg_triplet=None, max_objects=3, triplet_size=(384,384)):  
    
    sorted_objects = random.sample(range(1,num_objects+1),min(num_objects,max_objects))
    num_objects = len(sorted_objects)
      
    N_frames = torch.zeros(3, triplet_size[0], triplet_size[1], 3)
    N_masks = torch.zeros(3, triplet_size[0], triplet_size[1]) 
    
    if bg_triplet is not None:
        N_frames = bg_triplet
    
    for k in random.sample(range(num_objects),num_objects):
        
        obj_mask = (fg_mask == sorted_objects[k]).astype(np.uint8)      

        obj_rgb = obj_mask[:,:,None] * fg_frame
        
        rmin, rmax, cmin, cmax = bbox2(obj_mask)
        
        obj_rgb = obj_rgb[rmin : rmax, cmin : cmax]
        
        obj_mask = obj_mask[rmin : rmax, cmin : cmax]
        
        #pad with 100 pixels (50 by side)
        obj_rgb = img_pad(obj_rgb, 100)
        obj_mask = img_pad(obj_mask, 100)        
 
        
        global fg_rgb_transf
        global fg_mask_transf
        global fg_affine
        
        transf_obj_rgb = fg_rgb_transf(obj_rgb)
        transf_obj_mask = fg_mask_transf(obj_mask)
        
        
        obj_rgb_triplet = ChannelsLast()(transf_obj_rgb)
        obj_mask_triplet = ChannelsLast()(transf_obj_mask)[:,:,0]
        
        N_frames[0], N_masks[0] = random_paste(N_frames[0], N_masks[0], obj_rgb_triplet, obj_mask_triplet*(k+1), 100)
        
        for t in range(1,3):
            
            new_obj_rgb, new_obj_mask = fg_affine(transf_obj_rgb, transf_obj_mask)
            
            obj_rgb_triplet = ChannelsLast()(new_obj_rgb)
            obj_mask_triplet = ChannelsLast()(new_obj_mask).ceil()[:,:,0]
            
            N_frames[t], N_masks[t] = random_paste(N_frames[t], N_masks[t], obj_rgb_triplet, obj_mask_triplet*(k+1), 100)
    
    
    return N_frames, N_masks, num_objects

def random_paste(frame_canvas, mask_canvas, frame, mask, wpad, hpad=None):
    
    if hpad is None:
        hpad = wpad
        
    w_canvas, h_canvas = frame_canvas.shape[0], frame_canvas.shape[1]
    w_paste, h_paste = frame.shape[0], frame.shape[1]
    
    w_canvas_pos = random.randint(0, max(0, w_canvas - w_paste + wpad))
    h_canvas_pos = random.randint(0, max(0, h_canvas - h_paste + hpad))
    
    w_canvas_ini = max(0, w_canvas_pos - wpad//2)
    h_canvas_ini = max(0, h_canvas_pos - hpad//2)
    
    w_canvas_fin = min(w_canvas, w_canvas_pos + w_paste - wpad//2)
    h_canvas_fin = min(h_canvas, h_canvas_pos + h_paste - hpad//2)
    
    w_paste_pos_R = w_canvas_fin - w_canvas_pos
    h_paste_pos_R = h_canvas_fin - h_canvas_pos
    
    w_paste_pos_L = w_canvas_pos - w_canvas_ini
    h_paste_pos_L = h_canvas_pos - h_canvas_ini
    
    w_paste_ini = wpad//2 - w_paste_pos_L
    h_paste_ini = hpad//2 - h_paste_pos_L
    
    w_paste_fin = wpad//2 + w_paste_pos_R
    h_paste_fin = hpad//2 + h_paste_pos_R
    
    wci, wcf, hci, hcf = w_canvas_ini, w_canvas_fin, h_canvas_ini, h_canvas_fin
    wpi, wpf, hpi, hpf = w_paste_ini, w_paste_fin, h_paste_ini, h_paste_fin
    
    frame_canvas[wci:wcf, hci:hcf] = frame_canvas[wci:wcf, hci:hcf] * ~(mask[wpi:wpf, hpi:hpf, None] > 0)    
    frame_canvas[wci:wcf, hci:hcf] = frame_canvas[wci:wcf, hci:hcf] + frame[wpi:wpf, hpi:hpf] * (mask[wpi:wpf, hpi:hpf, None] > 0)
    
    mask_canvas[wci:wcf, hci:hcf] = mask_canvas[wci:wcf, hci:hcf] * ~(mask[wpi:wpf, hpi:hpf] > 0)    
    mask_canvas[wci:wcf, hci:hcf] = mask_canvas[wci:wcf, hci:hcf] + mask[wpi:wpf, hpi:hpf]    
    
    return frame_canvas, mask_canvas
 

def img_unpad(img, wpad, hpad=None):    
    
    if hpad is None:
        hpad = wpad
    
    w, h = img.shape[0], img.shape[1]    
    w_dif = np.floor((wpad)/2).astype(np.uint8)
    h_dif = np.floor((hpad)/2).astype(np.uint8)     
    
    return img[w_dif : w-w_dif, h_dif : h-h_dif]

def img_pad(img, wpad, hpad=None):    
    
    if hpad is None:
        hpad = wpad
    
    to_pad = (img.shape[0]+wpad, img.shape[1]+hpad)
    
    if len(img.shape) == 3:
        to_pad = to_pad+(0,)    
    
    shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(img.shape,to_pad)]
    shape_diffs = np.maximum(shape_diffs,0)
    pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
    
    return np.pad(img, pad_sizes, mode='constant')
    

def resize_keeping_aspect_ratio(img, new_size=(384,384)):    
    img_pil = Image.fromarray(img)
    img_w, img_h = img.shape[0], img.shape[1]
    (w,h) = new_size
        
    if (img_w - w < 0) or (img_h - h < 0):   
        if img_w <= img_h:
            wsize = w
            wpercent = (wsize/float(img_w))
            hsize = int((float(img_h)*float(wpercent)))                
        else:
            hsize = h
            wpercent = (hsize/float(img_h))
            wsize = int((float(img_w)*float(wpercent)))
        img_pil = img_pil.resize((hsize,wsize), Image.ANTIALIAS)
    
    return np.asarray(img_pil)
    

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax+1, cmin, cmax+1

def make_bg_triplet(bg=None, triplet_size=(384,384)):
    
    #get background image
    if bg is None:
        global coco_bg_cycle
        bg, _, _ = next(coco_bg_cycle)
    
    #resize background sides to be at least the same size of canvas sides
    bg = resize_keeping_aspect_ratio(bg, new_size=triplet_size)   
        
    bg_frames = torch.zeros(3, triplet_size[0], triplet_size[1], 3)
    
    global bg_transf
    
    for t in range(3):        
        bg_frames[t] = bg_transf(bg.copy())   
    
    
    return bg_frames



if __name__ == "__main__":
    
    
    print('inicio')
    
    
    # for aa in range(5):
    #     _, w, h = next(bg_pool)
    #     print('w: {}, h: {}'.format(w,h))
    
    # #get_background()
    # input("Press Enter to continue...")
    
    trainset = VOC_dataset(data_root='../rvos-master/databases', year='2012', imset='train')
    print('trainset instanciado, lenght: ', len(trainset))
    
    trainloader = data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=1)
    print('trainloader instanciado, lenght: ', len(trainloader))
    
    dataiter = iter(trainloader)
    print('dataiter instanciado')
    
    for aa in range(5):
        image, mask, num_objects, bg = dataiter.next()
        print('num objects: ', num_objects[0])
        
        
        
        #print('image: {}, label: {}, img_list: {}'.format(image,mask, img))        
        
        ff = plt.figure()
        ff.add_subplot(1,3,1)
        plt.imshow(image[0])
        ff.add_subplot(1,3,2)
        plt.imshow(mask[0])
        ff.add_subplot(1,3,3)
        plt.imshow(bg[0])
        plt.show(block=True) 
        input("Press Enter to continue...")











