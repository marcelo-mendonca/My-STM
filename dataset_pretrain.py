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
from torchsample.transforms import RandomAffine, RandomCrop, ToTensor, ChannelsFirst
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

fg_affine = RandomAffine(rotation_range=20, shear_range=10, zoom_range=(0.9, 1.1))

fg_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                     RangeNormalize(0, 1), fg_affine, ChannelsLast()])

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
    print('make triplet 2')
 
    
    teste = make_bg_triplet(bg, triplet_size)
    
    _, _, n = make_fg_triplet(fg_frame, fg_mask, num_objects)
    
    ff = plt.figure()
    ff.add_subplot(2,2,1)
    plt.imshow(bg)
    ff.add_subplot(2,2,2)
    plt.imshow(teste[0])
    ff.add_subplot(2,2,3)
    plt.imshow(teste[1])
    ff.add_subplot(2,2,4)
    plt.imshow(teste[2])
    plt.show(block=True) 
    input("Press Enter to continue...")  
    
    
    return        

def make_fg_triplet(fg_frame, fg_mask, num_objects, max_objects=3, triplet_size=(384,384)):
    
    w, h = triplet_size[0], triplet_size[1]
    
    print('FG_frame: ', fg_frame.shape)
    print('Num_objects antes: ', num_objects)    
    
    sorted_objects = random.sample(range(1,num_objects+1),min(num_objects,max_objects))
    print('sorted objects: ', sorted_objects)
    num_objects = len(sorted_objects)
    print('Num_objects depois: ', num_objects)
      
    N_frames = np.empty((3,)+triplet_size+(3,), dtype=np.uint8)
    N_masks = np.empty((3,)+triplet_size, dtype=np.uint8)
    print('N frames: ', N_frames.shape)
    print('N masks: ', N_masks.shape)    
    
    
    for k in range(num_objects):
        
        obj_mask = (fg_mask == sorted_objects[k]).astype(np.uint8)      

        obj_pixels = obj_mask[:,:,None] * fg_frame
        
        rmin, rmax, cmin, cmax = bbox2(obj_mask)
        
        obj_pixels = obj_pixels[rmin : rmax, cmin : cmax]
        
        obj_mask = obj_mask[rmin : rmax, cmin : cmax]
        
        print('obj pixels shape: {}, rmax-rmin: {}, cmax-cmin: {}'.format(obj_pixels.shape, rmax-rmin, cmax-cmin))
        
        size = (rmax-rmin+100, cmax-cmin+100)
        print('len size: ', len(size))

        
        print('len obj pix: ', len(obj_pixels.shape))
        pad_obj_pixels = img_pad(obj_pixels, 100)
        pad_obj_mask = img_pad(obj_mask, 100)
        
        
        size = (rmax-rmin, cmax-cmin)

        new_pad_obj_pixels = img_unpad(pad_obj_pixels, 100)
        new_pad_obj_mask = img_unpad(pad_obj_mask, 100)
        
        
        for t in range(3):
            
            frame_canvas = np.zeros(triplet_size+(3,), dtype=np.uint8)
            mask_canvas = np.zeros(triplet_size, dtype=np.uint8)
            
            

        
        ff = plt.figure()
        ff.add_subplot(2,2,1)
        plt.imshow(new_pad_obj_pixels)
        ff.add_subplot(2,2,2)
        plt.imshow(new_pad_obj_mask)
        ff.add_subplot(2,2,3)
        plt.imshow(pad_obj_pixels)
        ff.add_subplot(2,2,4)
        plt.imshow(pad_obj_mask)
        plt.show(block=True) 
        input("Press Enter to continue...")  
    
    fg_frames = torch.zeros(3, w, h, 3)
    fg_masks = torch.zeros(3, w, h)
    
    return fg_frames, fg_masks, num_objects

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
    
    print('make_triplet bg shape ANTES: ', bg.shape)
    
    #resize background sides to be at least the same size of canvas sides
    bg = resize_keeping_aspect_ratio(bg, new_size=triplet_size)   
    
    print('make_triplet bg shape DEPOIS: ', bg.shape)
        
    bg_frames = torch.zeros(3, triplet_size[0], triplet_size[1], 3)
    
    global bg_transf
    
    for t in range(3):        
        bg_frames[t] = bg_transf(bg.copy())   
    
    
    return bg_frames

def make_triplet(fg_frame, fg_mask, num_objects, bg=None, img_size=(384,384)):    
    
    #get background image
    if not bg.any():
        global coco_bg_cycle
        bg, bg_w, bg_h = next(coco_bg_cycle)
    else:
        bg_w, bg_h = bg.shape[0], bg.shape[1]
        
    print('make_triplet bg shape ANTES: ', bg.shape)
    
    #resize background sides to be at least 25% greater than canvas
    new_bg = Image.fromarray(bg)
    w, h = img_size[0], img_size[1]    
    if (bg_w - w < w/4) or (bg_h - h < h/4):   
        if bg_w <= bg_h:
            wsize = int(w + w/4)
            wpercent = (wsize/float(bg_w))
            hsize = int((float(bg_h)*float(wpercent)))                
        else:
            hsize = int(h + h/4)
            wpercent = (hsize/float(bg_h))
            wsize = int((float(bg_w)*float(wpercent)))
        new_bg = new_bg.resize((hsize,wsize), Image.ANTIALIAS)
        #bg = np.asarray(bg)   
    
    print('make_triplet bg shape DEPOIS: ', new_bg.size)
    
    
    print('Num_objects: ', num_objects)    
    N_masks = np.empty((3,)+fg_mask.shape, dtype=np.uint8)
    if num_objects > 3:        
        for k in range(3):
            N_masks[k] = fg_mask == k+1            
            num_objects = 3    
    else:
        for k in range(num_objects):
            N_masks[k] = fg_mask == k+1
    
    frame_canvas = Image.new(mode='RGB', size=img_size)
    mask_canvas = Image.new(mode='P', size=img_size)
    
    print('N_masks: ', N_masks.shape)
    print('frame_canvas: ', frame_canvas.size)
    print('mask_canvas: ', mask_canvas.size)
    
    frame_canvas.paste(new_bg)
    new_fg_frame = Image.fromarray(fg_frame)
    used_mask = Image.fromarray(N_masks[0]*255)
    frame_canvas.paste(new_fg_frame,(0,0),used_mask)
    new_bg.paste(frame_canvas, (100,100))
    
    
    ff = plt.figure()
    ff.add_subplot(2,2,1)
    plt.imshow(frame_canvas)
    ff.add_subplot(2,2,2)
    plt.imshow(new_bg)
    ff.add_subplot(2,2,3)
    plt.imshow(used_mask)
    ff.add_subplot(2,2,4)
    plt.imshow(new_fg_frame)
    plt.show(block=True) 
    input("Press Enter to continue...")
    
    
    # print('Pillow ANTES: ', bg.getpixel((100, 200)))
    # np_bg = np.asarray(bg)
    # print('Numpy: ', np_bg[200,100])
    # bg = Image.fromarray(np_bg)
    # print('Pillow DEPOIS: ', bg.getpixel((100, 200)))
    # bg_pix = bg.load()
    # print('Pillow INDEX: ', bg_pix[100, 200])
    
    
    #site sobre como usar o paste do pillow
    #https://note.nkmk.me/en/python-pillow-paste/
   
    return None




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











