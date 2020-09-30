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
from pycocotools.coco import COCO
import scipy.io
#from itertools import cycle


############################ Coco Background ################################  

class coco_background_loader():
    
    def __init__(self, data_root, imset='val', year='2017'):
        data_folder = 'COCO'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        set_year = '{}{}'.format(imset,year)
        self.image_dir = os.path.join(root, 'images',set_year)
        ann_dir = os.path.join(root, 'annotations')
        annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
        
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        self.num_imgs = len(self.img_ids)
        self.fixed_size = (384, 384)
    
    def get_background(self):        
        idx = random.randint(0, self.num_imgs-1)
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs(img_id)
        img_path = os.path.join(self.image_dir,img_data[0]['file_name'])       
        w = img_data[0]['width']
        h = img_data[0]['height']       
        
        return img_path, w, h
        



####################### Instanciate global objects########################## 
print('COCO Dataset----------------------')
coco_bg = coco_background_loader(data_root='../rvos-master/databases')
print('-------------------------Completed!')

bg_affine = RandomAffine(rotation_range=20, shear_range=10, zoom_range=(0.9, 1.1))
    
bg_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), RangeNormalize(0, 1), 
                     RandomBrightness(-0.1,0.1), bg_affine, RandomCrop(coco_bg.fixed_size), ChannelsLast()])

fg_affine = RandomAffine(rotation_range=90, shear_range=10, zoom_range=(0.5, 1.5))

fg_rgb_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                     RangeNormalize(0, 1), RandomBrightness(-0.1,0.1)])

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
               
        base_dir = os.path.join('VOCdevkit', 'VOC2012')
        voc_root = os.path.join(root, base_dir)        
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.mask_dir = os.path.join(voc_root, 'SegmentationObject')            
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, imset.rstrip('\n') + '.txt')
        self.fixed_size = (384, 384)
        self.k = 11
        
        with open(os.path.join(split_f), "r") as f:
            self.img_list = [x.strip() for x in f.readlines()]    
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'))
        mask = (mask != 255).astype(np.uint8) * mask
        num_objects = np.max(mask)
        
                
        N_frames, N_masks, num_objects = make_triplet(image, mask, num_objects, 
                                                     triplet_size=self.fixed_size,
                                                     k=self.k)
        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = [True, True, True]
        
        return N_frames, N_masks, num_objects, info


    def __len__(self):
        return len(self.img_list)
    
##################### ECSSD #######################
class ECSSD_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='', year=''):
        #../rvos-master/databases/ECSSD
        data_folder = 'ECSSD'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
               
        self.image_dir = os.path.join(root, 'images')
        self.mask_dir = os.path.join(root, 'ground_truth_mask')
        self.fixed_size = (384, 384)
        self.k = 11        
        
        self.img_list = ['{:04d}'.format(x) for x in range (1, 1001) ]
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
        mask = (mask/255).astype(np.uint8)
        num_objects = np.max(mask)
                
        N_frames, N_masks, num_objects = make_triplet(image, mask, num_objects, 
                                                     triplet_size=self.fixed_size,
                                                     k=self.k)        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = [True, True, True]
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### MSRA #######################
class MSRA_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='', year=''):
        #../rvos-master/databases/MSRA
        data_folder = 'MSRA'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        base_dir = os.path.join(root, 'MSRA10K_Imgs_GT')
        self.image_dir = os.path.join(base_dir, 'Imgs')
        self.mask_dir = os.path.join(base_dir, 'Imgs')
        self.fixed_size = (384, 384)
        self.k = 11        
        
        self.img_list = []
        for file in os.listdir(self.image_dir):
            if file.endswith(".jpg"):
                self.img_list.append(file.rstrip('.jpg'))          
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
        mask = (mask/255).astype(np.uint8)
        num_objects = np.max(mask)
                
        N_frames, N_masks, num_objects = make_triplet(image, mask, num_objects, 
                                                     triplet_size=self.fixed_size,
                                                     k=self.k)        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = [True, True, True]
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### SBD #######################
class SBD_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train.txt', year=''):
        #../rvos-master/databases/SBD
        data_folder = 'SBD'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        base_dir = os.path.join(root, 'benchmark_RELEASE', 'dataset')
        _imset_f = os.path.join(base_dir, imset)
        self.image_dir = os.path.join(base_dir, 'img')
        self.mask_dir = os.path.join(base_dir, 'inst')
        self.fixed_size = (384, 384)
        self.k = 11        
        
        self.img_list = []        
        
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                img_name = line.rstrip('\n')
                self.img_list.append(img_name)          
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".mat")
        
        image = np.array(Image.open(img_path).convert('RGB'))        
        mask = scipy.io.loadmat(mask_path)['GTinst']['Segmentation'][0,0]
        num_objects = np.max(mask)
                
        N_frames, N_masks, num_objects = make_triplet(image, mask, num_objects, 
                                                     triplet_size=self.fixed_size,
                                                     k=self.k)        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = [True, True, True]
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### COCO #######################
class COCO_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='val', year='2017'):
        data_folder = 'COCO'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        set_year = '{}{}'.format(imset,year)
        self.image_dir = os.path.join(root, 'images',set_year)
        ann_dir = os.path.join(root, 'annotations')
        self.annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
        
        self.coco = COCO(self.annFile)
        self.img_ids = self.coco.getImgIds()
        self.fixed_size = (384, 384)
        self.k = 11  
        
    def __getitem__(self, idx):        
        img_id = self.img_ids[idx]
        img_data = self.coco.loadImgs(img_id)
        img_path = os.path.join(self.image_dir,img_data[0]['file_name'])       
        w = img_data[0]['width']
        h = img_data[0]['height']
        
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        
        image = np.asarray(Image.open(img_path).convert('RGB'))
        mask = np.zeros((h,w))
        
        obj_areas = {}
        for i in range(len(anns)):  
            obj_areas[i] = anns[i]['area']
            
        obj_areas = {k: v for k, v in sorted(obj_areas.items(), key=lambda item: item[1], reverse=True)}
        
        obj_cc = 0
        for k, v in obj_areas.items(): 
            obj_cc += 1
            mask = np.maximum(self.coco.annToMask(anns[k])*obj_cc, mask)
            if obj_cc >= 5:
                break
        
        num_objects = obj_cc                
        N_frames, N_masks, num_objects = make_triplet(image, mask, num_objects, 
                                                     triplet_size=self.fixed_size,
                                                     k=self.k)        
        info = {}
        info['name'] = img_id
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = [True, True, True]
        
        return N_frames, N_masks, num_objects, info
    
    def __len__(self):
        return len(self.img_ids)

############################ AUX FUNCTIONS ################################

def make_triplet(fg_frame, fg_mask, num_objects, triplet_size=(384,384), k=11):      
    
    bg_triplet = make_bg_triplet(triplet_size=triplet_size)
    
    N_frames, N_masks, num_objects = make_fg_triplet(fg_frame, fg_mask, num_objects, 
                                                     bg_triplet, triplet_size=triplet_size)    
    
    N_frames = N_frames.permute(3,0,1,2).float()
    
    num_objects = torch.LongTensor([num_objects])
    
    N_masks = All_to_onehot(N_masks, k).float()  
    
    return N_frames, N_masks, num_objects

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
        
        try:        
            rmin, rmax, cmin, cmax = bbox2(obj_mask)        
            obj_rgb = obj_rgb[rmin : rmax, cmin : cmax]        
            obj_mask = obj_mask[rmin : rmax, cmin : cmax]
        except:
            pass
        
        #pad with 100 pixels (50 by side)
        obj_rgb = img_pad(obj_rgb, 100)
        obj_mask = img_pad(obj_mask, 100)   
        
        global fg_rgb_transf
        global fg_mask_transf
        global fg_affine
        
        transf_obj_rgb = fg_rgb_transf(obj_rgb)
        transf_obj_mask = fg_mask_transf(obj_mask)
        
        obj_rgb_triplet = ChannelsLast()(transf_obj_rgb)
        obj_mask_triplet = transf_obj_mask[0]
        
        obj_rgb_triplet, obj_mask_triplet = random_flip(obj_rgb_triplet, obj_mask_triplet)
        
        N_frames[0], N_masks[0] = random_paste(N_frames[0], N_masks[0], obj_rgb_triplet, obj_mask_triplet*(k+1), 100)
        
        for t in range(1,3):
            
            new_obj_rgb, new_obj_mask = fg_affine(transf_obj_rgb, transf_obj_mask)
            
            obj_rgb_triplet = ChannelsLast()(new_obj_rgb)
            obj_mask_triplet = new_obj_mask[0].ceil()
            
            obj_rgb_triplet, obj_mask_triplet = random_flip(obj_rgb_triplet, obj_mask_triplet)
            
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

def random_flip(frame, mask, probh=20, probv=20):
    
    if random.randint(1,100) <= probv:        
        frame = torch.flip(frame, [0])
        mask = torch.flip(mask, [0])
    
    if random.randint(1,100) <= probh:        
        frame = torch.flip(frame, [1])
        mask = torch.flip(mask, [1])    
    
    return frame, mask
 

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
    

def resize_to_minimum_size(img, min_size=(384,384)):        
    img_w, img_h = img.size[0], img.size[1]
    (w,h) = min_size

    if (img_w - w < 0) or (img_h - h < 0):   
        if img_w <= img_h:
            wsize = w
            wpercent = (wsize/float(img_w))
            hsize = int((float(img_h)*float(wpercent)))                
        else:
            hsize = h
            wpercent = (hsize/float(img_h))
            wsize = int((float(img_w)*float(wpercent)))
        img = img.resize((wsize,hsize), Image.ANTIALIAS)
    
    return np.asarray(img)
    

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax+1, cmin, cmax+1

def make_bg_triplet(bg_path=None, triplet_size=(384,384)):
    
    #get background image
    if bg_path is None:
        global coco_bg
        bg_path, _, _ = coco_bg.get_background()
        #bg_path = bg_path[0]    
    bg_img = Image.open(bg_path).convert('RGB')
    
    #resize background sides to be at least the same size of canvas sides
    bg = resize_to_minimum_size(bg_img, min_size=triplet_size)   
        
    bg_triplet = torch.zeros(3, triplet_size[0], triplet_size[1], 3)
    
    global bg_transf
    
    for t in range(3):        
        bg_triplet[t] = bg_transf(bg.copy())    
    
    return bg_triplet

def To_onehot(mask, K):
    M = torch.zeros(K, mask.shape[0], mask.shape[1]).int()
    for k in range(K):
        M[k] = (mask == k).int()
    return M # M:  (11, 384, 384)

def All_to_onehot(masks, K):
    # k = 11
    # masks:  (3, 384, 384) -> 3 máscaras do train_triplet
    Ms = torch.zeros(K, masks.shape[0], masks.shape[1], masks.shape[2]).int()
    for n in range(masks.shape[0]): #n -> 0, 1, 2
        Ms[:,n] = To_onehot(masks[n], K)
        
    return Ms # Ms:  (11, 3, 384, 384)


if __name__ == "__main__":
    
    
    print('inicio') 
                
    #input("Press Enter to continue...") 

    
    VOC_trainset = VOC_dataset(data_root='../rvos-master/databases', year='2012', imset='train')
    
    ECSSD_trainset = ECSSD_dataset(data_root='../rvos-master/databases')
    
    MSRA_trainset = MSRA_dataset(data_root='../rvos-master/databases')
    
    SBD_trainset = SBD_dataset(data_root='../rvos-master/databases')
    
    COCO_trainset = COCO_dataset(data_root='../rvos-master/databases')
    
    trainloader = data.DataLoader(COCO_trainset, batch_size=1,
                                          shuffle=True, num_workers=0)
    print('trainloader instanciado, lenght: ', len(trainloader))
    
    dataiter = iter(trainloader)
    
    for cc in range(10):
        
        image, mask, n_obj, _ = dataiter.next()
        
        #image, mask, n_obj = dataiter.next()
        
        # ff = plt.figure()
        # ff.add_subplot(1,2,1)
        # plt.imshow(image[0])
        # ff.add_subplot(1,2,2)
        # plt.imshow(mask[0])
        # plt.show(block=True)
        
    
        N_frames = image[0].permute(1, 2, 3, 0)
        N_masks = mask[0].permute(1, 2, 3, 0)
        num_objects = n_obj[0].item()
            
        # print('N_frames: {}, {}'.format(N_frames.shape, N_frames.dtype))    
        # print('N_masks: {}, {}'.format(N_masks.shape, N_masks.dtype))    
        print('num_objects: {}'.format(num_objects))
        # # N_frames: torch.Size([1, 3, 3, 384, 384]), torch.float32
        # # N_masks: torch.Size([1, 11, 3, 384, 384]), torch.int32
        # # num_objects: tensor([[1]]), torch.int64
        
       
    
        for hh in range(1):
            print('Mask CH: ', hh)
            ff = plt.figure()
            ff.add_subplot(2,3,1)
            plt.imshow(N_frames[0])
            ff.add_subplot(2,3,2)
            plt.imshow(N_frames[1])
            ff.add_subplot(2,3,3)
            plt.imshow(N_frames[2])
            ff.add_subplot(2,3,4)
            plt.imshow(N_masks[0,:,:,hh])
            ff.add_subplot(2,3,5)
            plt.imshow(N_masks[1,:,:,hh])
            ff.add_subplot(2,3,6)
            plt.imshow(N_masks[2,:,:,hh])
            plt.show(block=True) 
            
        input("Press Enter to continue...") 


###############################################################################

#coco_bg_loader = data.DataLoader(coco_bg, batch_size=1, shuffle=True, num_workers=1)
#coco_bg_cycle = cycle(coco_bg_loader)

# class coco_background(data.Dataset):
    
#     def __init__(self, data_root, imset='val', year='2017'):
#         data_folder = 'COCO'
#         root = os.path.join(data_root, data_folder)
#         if not os.path.isdir(root):
#             raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
#         self.root = root 
#         set_year = '{}{}'.format(imset,year)
#         self.image_dir = os.path.join(self.root, 'images',set_year)
#         ann_dir = os.path.join(self.root, 'annotations')
#         self.annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
        
#         self.coco = COCO(self.annFile)
#         self.img_ids = self.coco.getImgIds()
#         self.fixed_size = (384, 384)   
        
#     def __getitem__(self, idx):
        
#         img_id = self.img_ids[idx]
#         img_data = self.coco.loadImgs(img_id)
#         img_path = os.path.join(self.image_dir,img_data[0]['file_name'])       
#         w = img_data[0]['width']
#         h = img_data[0]['height']       
        
#         return img_path, w, h
    
#     def __len__(self):
#         return len(self.img_ids)



