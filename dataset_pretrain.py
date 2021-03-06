#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:13:38 2020

@author: marcelo
"""
from config import cfg
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import random
from torchsample.transforms import RandomAffine, RandomCrop, ToTensor, ChannelsFirst, AddChannel
from torchsample.transforms import ChannelsLast, Compose, RandomBrightness, RangeNormalize, TypeCast
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import scipy.io


############################ Coco Background ################################  

class coco_img_loader():    
    def __init__(self, data_root, imsets=['val'], year='2017', fixed_size={'val': (384,384)}):
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
            
        self.image_dir = {}
        self.imsets = imsets
        self.coco = {}
        self.img_ids = {}
        self.num_imgs = {}
        self.fixed_size = {}
        
        for imset in self.imsets:        
            set_year = '{}{}'.format(imset,year)
            self.image_dir[imset] = os.path.join(data_root, 'images',set_year)
            ann_dir = os.path.join(data_root, 'annotations')
            annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
            
            self.coco[imset] = COCO(annFile)
            self.img_ids[imset] = self.coco[imset].getImgIds()
            self.num_imgs[imset] = len(self.img_ids[imset])
            self.fixed_size[imset] = fixed_size[imset]
    
    def get_random_img(self, imset):        
        idx = random.randint(0, self.num_imgs[imset]-1)
        img_id = self.img_ids[imset][idx]
        img_data = self.coco[imset].loadImgs(img_id)
        img_path = os.path.join(self.image_dir[imset],img_data[0]['file_name'])       
        w = img_data[0]['width']
        h = img_data[0]['height']       
        
        return img_path, w, h
    
    def get_loader(self, imset):
        return self.coco[imset], self.img_ids[imset], self.num_imgs[imset], self.image_dir[imset]


##################### Instanciate global COCO loader ########################
if cfg.PRETRAIN:
    print('COCO data loader-----------------')
    coco_bg = coco_img_loader(data_root=cfg.DATA_COCO, imsets=['train','val'], 
                                     fixed_size={'train': cfg.TRAIN_IMG_SIZE, 'val': cfg.VAL_IMG_SIZE})
    print('-------------------------Completed!')

################### Pascal VOC #####################
class VOC_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train', year='2012', fixed_size=(384,384)):
        #../rvos-master/databases/VOC/VOCdevkit/VOC2012
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
                       
        self.imset = imset #VOC has train and val sets
        self.image_dir = os.path.join(data_root, 'JPEGImages')
        self.mask_dir = os.path.join(data_root, 'SegmentationObject')            
        splits_dir = os.path.join(data_root, 'ImageSets/Segmentation')
        _imset = 'trainval' if imset == 'val' else imset
        split_f = os.path.join(splits_dir, _imset + '.txt')
        self.fixed_size = fixed_size
        self.k = 11
        
        with open(split_f, "r") as f:
            self.img_list = [x.strip() for x in f.readlines()]    
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'))
        mask = (mask != 255).astype(np.uint8) * mask
        num_objects = np.max(mask)
        
                
        N_frames, N_masks, num_objects, is_valid = make_triplet(image, mask, num_objects, 
                                                                triplet_size=self.fixed_size,
                                                                k=self.k, imset=self.imset)     
        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = is_valid
        
        return N_frames, N_masks, num_objects, info


    def __len__(self):
        return len(self.img_list)
    
##################### ECSSD #######################
class ECSSD_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train', year='', fixed_size=(384,384)):
        #../rvos-master/databases/ECSSD
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
               
        self.imset = imset #ECSSD has only train set
        self.image_dir = os.path.join(data_root, 'images')
        self.mask_dir = os.path.join(data_root, 'ground_truth_mask')
        self.fixed_size = fixed_size
        self.k = 11        
        
        self.img_list = ['{:04d}'.format(x) for x in range (1, 1001) ]
    
    def __getitem__(self, idx):        
        img_path = os.path.join(self.image_dir, self.img_list[idx] + ".jpg")
        mask_path = os.path.join(self.mask_dir, self.img_list[idx] + ".png")
        
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('P'), dtype=np.uint8)
        mask = (mask/255).astype(np.uint8)
        num_objects = np.max(mask)
                
        N_frames, N_masks, num_objects, is_valid = make_triplet(image, mask, num_objects, 
                                                                triplet_size=self.fixed_size,
                                                                k=self.k, imset=self.imset)     
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = is_valid
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### MSRA #######################
class MSRA_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train', year='', fixed_size=(384,384)):
        #../rvos-master/databases/MSRA/MSRA10K_Imgs_GT
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
        
        self.imset = imset #MSRA has only trainset
        self.image_dir = os.path.join(data_root, 'Imgs')
        self.mask_dir = os.path.join(data_root, 'Imgs')
        self.fixed_size = fixed_size
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
                
        N_frames, N_masks, num_objects, is_valid = make_triplet(image, mask, num_objects, 
                                                                triplet_size=self.fixed_size,
                                                                k=self.k, imset=self.imset)        
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = is_valid
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### SBD #######################
class SBD_dataset(data.Dataset):
    
    def __init__(self, data_root, imset='train', year='', fixed_size=(384,384)):
        #../rvos-master/databases/SBD/benchmark_RELEASE/dataset
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))

        self.imset = imset # SBD has train and val sets
        _imset_f = os.path.join(data_root, imset + '.txt')
        self.image_dir = os.path.join(data_root, 'img')
        self.mask_dir = os.path.join(data_root, 'inst')
        self.fixed_size = fixed_size
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
                
        N_frames, N_masks, num_objects, is_valid = make_triplet(image, mask, num_objects, 
                                                                triplet_size=self.fixed_size,
                                                                k=self.k, imset=self.imset)   
        info = {}
        info['name'] = self.img_list[idx]
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = is_valid
        
        return N_frames, N_masks, num_objects, info

    def __len__(self):
        return len(self.img_list)

##################### COCO #######################
class COCO_dataset(data.Dataset):
    #../rvos-master/databases/COCO    
    def __init__(self, data_root, imset='train', year='2017', fixed_size=(384,384)):
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))        
        
        self.imset = imset # COCO has train and val sets
        set_year = '{}{}'.format(imset,year)
        self.image_dir = os.path.join(data_root, 'images',set_year)
        ann_dir = os.path.join(data_root, 'annotations')
        self.annFile = os.path.join(ann_dir, 'instances_{}.json'.format(set_year))
        
        global coco_bg
        try:
            self.coco, self.img_ids, _, _ = coco_bg.get_loader(self.imset)
        except:
            self.coco = COCO(self.annFile)
            self.img_ids = self.coco.getImgIds()
        self.fixed_size = fixed_size
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
        N_frames, N_masks, num_objects, is_valid = make_triplet(image, mask, num_objects, 
                                                                triplet_size=self.fixed_size,
                                                                k=self.k, imset=self.imset)        
        info = {}
        info['name'] = img_id
        info['num_frames'] = N_frames.shape[1]
        info['valid_samples'] = is_valid
        
        return N_frames, N_masks, num_objects, info
    
    def __len__(self):
        return len(self.img_ids)

############################ AUX FUNCTIONS ################################

def make_triplet(fg_frame, fg_mask, num_objects, triplet_size=(384,384), k=11, imset='val'):

    num_objects = max(1, num_objects) #ensure at least one object
    
    bg_triplet = make_bg_triplet(triplet_size=triplet_size, imset=imset)
    
    N_frames, N_masks, num_objects = make_fg_triplet(fg_frame, fg_mask, num_objects, 
                                                     bg_triplet, triplet_size=triplet_size)   

    N_frames = N_frames.permute(3,0,1,2).float()
    
    num_objects = torch.LongTensor([num_objects])
    
    N_masks = All_to_onehot(N_masks, k).float()  
    
    return N_frames, N_masks, num_objects, validate_samples(N_masks, num_objects)

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
        
        #global fg_rgb_transf
        #global fg_mask_transf
        #global fg_affine
        
        fg_affine = RandomAffine(rotation_range=90, shear_range=10, zoom_range=(0.5, 1.5))

        fg_rgb_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                             RangeNormalize(0, 1), RandomBrightness(-0.1,0.1)])
        
        fg_mask_transf = Compose([ToTensor(), AddChannel(axis=0), TypeCast('float'), 
                                  RangeNormalize(0, 1)])
        
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

def make_bg_triplet(bg_path=None, triplet_size=(384,384), imset='val'):    
    #get background image
    if bg_path is None:
        global coco_bg
        bg_path, _, _ = coco_bg.get_random_img(imset)
            
    bg_img = Image.open(bg_path).convert('RGB')
    
    #resize background sides to be at least the same size of canvas sides
    bg = resize_to_minimum_size(bg_img, min_size=triplet_size)   
        
    bg_triplet = torch.zeros(3, triplet_size[0], triplet_size[1], 3)
    
    bg_affine = RandomAffine(rotation_range=20, shear_range=10, zoom_range=(0.9, 1.1))
    
    bg_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), RangeNormalize(0, 1), 
                         RandomBrightness(-0.1,0.1), bg_affine, RandomCrop(coco_bg.fixed_size[imset]), ChannelsLast()])
    
    #global bg_transf
    
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

def validate_samples(N_masks, num_objects):
    
    if num_objects <= 0:
        return False
    
    if not torch.max(N_masks[0]) > 0:
        return False
    elif not (torch.max(N_masks[1]) > 0 or torch.max(N_masks[2]) > 0):
        return False
    
    return True


if __name__ == "__main__":
    
    
    print('inicio')
    
    
    import time
     
    
    FIXED_SIZE = (150, 150) 
    
    # Train
    VOC_trainset = VOC_dataset(data_root=cfg.DATA_VOC, fixed_size=FIXED_SIZE)    
    ECSSD_trainset = ECSSD_dataset(data_root=cfg.DATA_ECSSD, fixed_size=FIXED_SIZE)    
    MSRA_trainset = MSRA_dataset(data_root=cfg.DATA_MSRA, fixed_size=FIXED_SIZE)    
    SBD_trainset = SBD_dataset(data_root=cfg.DATA_SBD, fixed_size=FIXED_SIZE)    
    COCO_trainset = COCO_dataset(data_root=cfg.DATA_COCO, fixed_size=FIXED_SIZE)    
    trainset = data.ConcatDataset([VOC_trainset]+[ECSSD_trainset]+[MSRA_trainset]+[SBD_trainset]+[COCO_trainset])    
    dataloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    
    # Val
    # VOC_valset = VOC_dataset(data_root=cfg.DATA_VOC, fixed_size=FIXED_SIZE, imset='val')    
    # SBD_valset = SBD_dataset(data_root=cfg.DATA_SBD, fixed_size=FIXED_SIZE, imset='val')
    # COCO_valset = COCO_dataset(data_root=cfg.DATA_COCO, fixed_size=FIXED_SIZE, imset='val')
    # valset = data.ConcatDataset([VOC_valset]+[SBD_valset]+[COCO_valset])
    # dataloader = data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    
    print('dataloader lenght: ', len(dataloader))    
    # trainloader lenght:  139249
    # valloader lenght:  10770
    
    dataiter = iter(dataloader)
    
    #image, mask, n_obj, info = dataiter.next()
    boa = 0
    ruim = 0
    t0= time.process_time()
    for cc in range(200):
        _, _, _, info = dataiter.next()
        
        t1 = time.process_time() - t0
        print("{}) = {}, Time: {:.3f}".format(cc, info['valid_samples'], t1)) # CPU seconds elapsed (floating point)
        t0 = t1
        
        if info['valid_samples']:
            boa +=1
        else:
            ruim +=1
    
    print('boas {}, ruins {}'.format(boa, ruim))
        
        #image, mask, n_obj = dataiter.next()
        
        # ff = plt.figure()
        # ff.add_subplot(1,2,1)
        # plt.imshow(image[0])
        # ff.add_subplot(1,2,2)
        # plt.imshow(mask[0])
        # plt.show(block=True)
    input("Press Enter to continue...") 
    
    # if False:
    
    #     N_frames = image[0].permute(1, 2, 3, 0)
    #     N_masks = mask[0].permute(1, 2, 3, 0)
    #     num_objects = n_obj[0].item()
            
    #     # print('N_frames: {}, {}'.format(N_frames.shape, N_frames.dtype))    
    #     # print('N_masks: {}, {}'.format(N_masks.shape, N_masks.dtype))    
    #     print('num_objects: {}'.format(num_objects))
    #     # # N_frames: torch.Size([1, 3, 3, 384, 384]), torch.float32
    #     # # N_masks: torch.Size([1, 11, 3, 384, 384]), torch.int32
    #     # # num_objects: tensor([[1]]), torch.int64
        
       
    
    #     for hh in range(1):
    #         print('Mask CH: ', hh)
    #         ff = plt.figure()
    #         ff.add_subplot(2,3,1)
    #         plt.imshow(N_frames[0])
    #         ff.add_subplot(2,3,2)
    #         plt.imshow(N_frames[1])
    #         ff.add_subplot(2,3,3)
    #         plt.imshow(N_frames[2])
    #         ff.add_subplot(2,3,4)
    #         plt.imshow(N_masks[0,:,:,hh])
    #         ff.add_subplot(2,3,5)
    #         plt.imshow(N_masks[1,:,:,hh])
    #         ff.add_subplot(2,3,6)
    #         plt.imshow(N_masks[2,:,:,hh])
    #         plt.show(block=True) 
            
    #     input("Press Enter to continue...") 



