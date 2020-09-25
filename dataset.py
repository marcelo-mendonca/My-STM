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
from torchsample.transforms import RandomAffine, ToTensor, ChannelsFirst, AddChannel
from torchsample.transforms import ChannelsLast, Compose, RandomBrightness, RangeNormalize, TypeCast
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters.rank import modal
from dataset_pretrain import All_to_onehot


############################ YOUTUBE TRAIN ################################
class Youtube_MO_Train(data.Dataset):    
    def __init__(self, data_root, imset='train-train-meta.json', resolution='480p', single_object=False, frame_skip=5):
        #../rvos-master/databases/YouTubeVOS/train
        data_folder = 'YouTubeVOS/train'
        root = os.path.join(data_root, data_folder)        
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        mask_dir = os.path.join(root, 'Annotations')
        image_dir = os.path.join(root, 'JPEGImages')
        _imset_dir = os.path.join(root)
        _imset_f = os.path.join(_imset_dir, imset)
        
        self.videos = []
        self.num_frames = {}
        self.frame_paths = {}
        self.mask_paths = {}
        self.num_objects = {}
        self.train_triplets = {}
        self.frame_skip = frame_skip
        self.fixed_size = (384, 384) #(100, 100)
        self.frame_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                                     RangeNormalize(0, 1), RandomBrightness(-0.1, 0.1)])
        self.mask_transf = Compose([ToTensor(), AddChannel(axis=0), TypeCast('float')])
        self.affine_transf = RandomAffine(rotation_range=15, shear_range=10, 
                                          zoom_range=(0.95, 1.05))
        idx = 0        
        with open(_imset_f) as json_file:
            json_data = edict(json.load(json_file))
            for _video in json_data.videos.keys():
                self.videos.append(_video)
                self.frame_paths[_video] = glob.glob(os.path.join(image_dir, _video, '*.jpg'))
                self.num_frames[_video] = len(self.frame_paths[_video])
                self.mask_paths[_video] = glob.glob(os.path.join(mask_dir, _video, '*.png'))
                self.num_objects[_video] = len(json_data.videos[_video].objects)
                
                #starts from 2 so that there are at least 2 frames before
                for f in range(2, self.num_frames[_video]):
                    self.train_triplets[idx] = (_video, f)
                    idx += 1
        self.K = 11
        self.single_object = single_object        

    def __getitem__(self, index):
        video = self.train_triplets[index][0]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]

        N_frames = torch.zeros(3, self.fixed_size[0], self.fixed_size[1], 3)
        N_masks = torch.zeros(3, self.fixed_size[0], self.fixed_size[1]) 
        #N_frames:  (n, w, h, c) -> num frames, widith, height, channel
        #N_masks:  (n, w, h)
        
        train_triplet =  Skip_frames(self.train_triplets[index][1], self.frame_skip, self.num_frames[video])
        #train_triplet = (t1, t2, t3) -> sequence for getting frames at t1 < t2 < t3
        
        valid_samples = [False, False, False]
        
        for idx, f in enumerate(train_triplet):
            frame_path = self.frame_paths[video][f]
            frame, coord = resize_keeping_aspect_ratio(Image.open(frame_path).convert('RGB'), self.fixed_size)
            #frame:  numpy (w>=fixed_size[0], h>=fixed_size[1], c)
            
            try:
                mask_path = self.mask_paths[video][f]
                mask, _ = resize_keeping_aspect_ratio(Image.open(mask_path).convert('P'), self.fixed_size, coord)
            except:
                mask = np.zeros(self.fixed_size, dtype=np.uint8)
            #masks:  numpy (w>=fixed_size, h>=fixed_size)
            
            frame_transf, mask_transf = self.affine_transf(self.frame_transf(frame.copy()), self.mask_transf(mask.copy()))
            #frame_transf:  torch (c, w, h) -> c=3
            #mask_transf:  torch (c, w, h) -> c=1
            
            new_frame = ChannelsLast()(frame_transf)
            new_mask = torch.from_numpy(modal(np.array(mask_transf[0]).astype(np.uint8), disk(5))) # modal function clean up some noise caused by affine_transf
            #new_frame:  torch (w, h, c)
            #new_mask:  torch (w, h)
            
            N_frames[idx], N_masks[idx] = random_crop(N_frames[idx], N_masks[idx], new_frame, new_mask, 0)
            #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
            #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            
            valid_samples[idx] = (torch.max(N_masks[idx]) > 0).item()
        
        info['valid_samples'] = valid_samples            
        N_frames = N_frames.permute(3, 0, 1, 2)
        if self.single_object:
            N_masks = (N_masks > 0.5).int() * (N_masks < 255).int()
            N_masks = All_to_onehot(N_masks, self.K).float()
            num_objects = torch.LongTensor([int(1)])
            return N_frames, N_masks, num_objects, info
        else:
            N_masks = All_to_onehot(N_masks, self.K).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return N_frames, N_masks, num_objects, info
            #N_frames: torch (c, n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            #N_masks: torch (k, n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            #num_objects:  tensor([[num_objects]]) -> torch.int
            #info:  {'name', 'num_frames', 'valid_samples'}
            #Obs. dataloader adds batch dimension in position 0 for each returned tensor          
    
    def __len__(self):
        return len(self.train_triplets)
    

############################ DAVIS TRAIN ################################
class DAVIS_MO_Train(data.Dataset):    
    def __init__(self, data_root, imset='2017/train.txt', resolution='480p', single_object=False, frame_skip=5):
        #../rvos-master/databases/DAVIS2017
        data_folder = 'DAVIS2017'
        root = os.path.join(data_root, data_folder)        
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.train_triplets = {}
        self.frame_skip = frame_skip
        self.fixed_size = (384, 384) #(100, 100)
        self.frame_transf = Compose([ToTensor(), TypeCast('float'), ChannelsFirst(), 
                                     RangeNormalize(0, 1), RandomBrightness(-0.1, 0.1)])
        self.mask_transf = Compose([ToTensor(), AddChannel(axis=0), TypeCast('float')])
        self.affine_transf = RandomAffine(rotation_range=15, shear_range=10, 
                                          zoom_range=(0.95, 1.05))
        idx = 0
        
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                
                #starts from 2 so that there are at least 2 frames before
                for f in range(2, self.num_frames[_video]):
                    self.train_triplets[idx] = (_video, f)
                    idx += 1
        self.K = 11
        self.single_object = single_object

    def __getitem__(self, index):
        video = self.train_triplets[index][0]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        first_mask = np.array(Image.open(os.path.join(self.mask_dir, video, '00000.png')).convert("P"))
        self.num_objects[video] = np.max(first_mask)
        
        N_frames = torch.zeros(3, self.fixed_size[0], self.fixed_size[1], 3)
        N_masks = torch.zeros(3, self.fixed_size[0], self.fixed_size[1]) 
        #N_frames:  (n, w, h, c) -> num frames, widith, height, channel
        #N_masks:  (n, w, h)
        
        train_triplet =  Skip_frames(self.train_triplets[index][1], self.frame_skip, self.num_frames[video])
        #train_triplet = (t1, t2, t3) -> sequence for getting frames at t1 < t2 < t3
        
        valid_samples = [False, False, False]
        
        for idx, f in enumerate(train_triplet):
            frame_path = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            frame, coord = resize_keeping_aspect_ratio(Image.open(frame_path).convert('RGB'), self.fixed_size)
            #frame:  numpy (w>=fixed_size[0], h>=fixed_size[1], c)
            
            try:
                mask_path = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
                mask, _ = resize_keeping_aspect_ratio(Image.open(mask_path).convert('P'), self.fixed_size, coord)
            except:
                mask = np.zeros(self.fixed_size, dtype=np.uint8)
            #masks:  numpy (w>=fixed_size, h>=fixed_size)
                
            frame_transf, mask_transf = self.affine_transf(self.frame_transf(frame.copy()), self.mask_transf(mask.copy()))
            #frame_transf:  torch (c, w, h) -> c=3
            #mask_transf:  torch (c, w, h) -> c=1
            
            new_frame = ChannelsLast()(frame_transf)
            new_mask = torch.from_numpy(modal(np.array(mask_transf[0]).astype(np.uint8), disk(5))) # modal function clean up some noise caused by affine_transf
            #new_frame:  torch (w, h, c)
            #new_mask:  torch (w, h)
            
            N_frames[idx], N_masks[idx] = random_crop(N_frames[idx], N_masks[idx], new_frame, new_mask, 0)
            #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
            #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            
            valid_samples[idx] = (torch.max(N_masks[idx]) > 0).item()
        
        info['valid_samples'] = valid_samples
        N_frames = N_frames.permute(3, 0, 1, 2)
        if self.single_object:
            N_masks = (N_masks > 0.5).int() * (N_masks < 255).int()
            N_masks = All_to_onehot(N_masks, self.K).float()
            num_objects = torch.LongTensor([int(1)])
            return N_frames, N_masks, num_objects, info
        else:
            N_masks = All_to_onehot(N_masks, self.K).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return N_frames, N_masks, num_objects, info
            #N_frames: torch (c, n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            #N_masks: torch (k, n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
            #num_objects:  tensor([[num_objects]]) -> torch.int
            #info:  {'name', 'num_frames', 'valid_samples'}
            #Obs. dataloader adds batch dimension in position 0 for each returned tensor  
    
    def __len__(self):
        return len(self.train_triplets)
            
############################ YOUTUBE VALIDATION ###############################
class Youtube_MO_Val(data.Dataset):    
    def __init__(self, data_root, imset='train-val-meta.json', resolution='480p', single_object=False):
        #../rvos-master/databases/YouTubeVOS/train
        data_folder = 'YouTubeVOS/train'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations')
        self.mask480_dir = os.path.join(root, 'Annotations')
        self.image_dir = os.path.join(root, 'JPEGImages')
        _imset_dir = os.path.join(root)
        _imset_f = os.path.join(_imset_dir, imset)
        
        self.videos = []
        self.num_frames = {}
        self.frame_paths = {}
        self.mask_paths = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        self.batch_frames = {}
        self.fixed_size = (100, 100) #(384,384)
        idx = 0 
        
        with open(_imset_f) as json_file:
            json_data = edict(json.load(json_file))
            for _video in json_data.videos.keys():
                self.videos.append(_video)
                self.frame_paths[_video] = glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))
                self.num_frames[_video] = len(self.frame_paths[_video])
                self.mask_paths[_video] = glob.glob(os.path.join(self.mask_dir, _video, '*.png'))
                self.num_objects[_video] = len(json_data.videos[_video].objects)
                self.shape[_video] = self.fixed_size
                self.size_480p[_video] = self.fixed_size
                
                #starts from 0 to get all frames in sequence
                for f in range(self.num_frames[_video]):
                    self.batch_frames[idx] = (_video, f)
                    idx += 1
        
        self.K = 11
        self.single_object = single_object

    def __getitem__(self, index):
        video = self.batch_frames[index][0]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)
        #N_frames:  (1, 384, 384, 3) -> frames, w, h, ch
        #N_masks:  (1, 384, 384) -> frames, w, h
        
        frame_idx = self.batch_frames[index][1]
        img_file = self.frame_paths[video][frame_idx]
        N_frames[0], coord = Crop_frames(Image.open(img_file).convert('RGB'), self.fixed_size)
        
        try:
            mask_file = self.mask_paths[video][frame_idx]
            N_masks[0], coord = Crop_frames(Image.open(mask_file).convert('P'), self.fixed_size, coord)
        except:
            N_masks[0] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        #Fs:  torch.Size([3, 3, 480, 854]) -> canais, frames, linhas, colunas
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(All_to_onehot(N_masks, self.K).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(All_to_onehot(N_masks, self.K).copy()).float()
            #Ms:  torch.Size([11, 3, 480, 854]) -> canais, frames, linhas, colunas
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info            
            #Chega pelo dataloader lá na chamada com uma dimensao extra:
            #Fs:  torch.Size([1, 3, 3, 480, 854])
            #Ms:  torch.Size([1, 11, 3, 480, 854])
            #num_objects:  tensor([[2]])
            #info:  {'name': ['bike-packing'], 'num_frames': tensor([69]), 
            #   'size_480p': [tensor([480]), tensor([854])]}
    
    def __len__(self):
        return len(self.batch_frames)
    
############################ DAVIS VALIDATION ################################
class DAVIS_MO_Val(data.Dataset):
    def __init__(self, data_root, imset='2017/val.txt', resolution='480p', single_object=False):
        #../rvos-master/databases/DAVIS2017
        data_folder = 'DAVIS2017'
        root = os.path.join(data_root, data_folder)
        if not os.path.isdir(root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(root))
        
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        self.batch_frames = {}
        self.fixed_size = (100, 100) #(384,384)
        idx = 0
        
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                #_mask:  (480, 854)
                self.num_objects[_video] = np.max(_mask)
                #self.shape[_video] = np.shape(_mask)
                self.shape[_video] = self.fixed_size
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                #_mask480:  (480, 854)
                #self.size_480p[_video] = np.shape(_mask480)
                self.size_480p[_video] = self.fixed_size
                
                #starts from 0 to get all frames in sequence
                for f in range(self.num_frames[_video]):
                    self.batch_frames[idx] = (_video, f)
                    idx += 1
        self.K = 11
        self.single_object = single_object

    def __getitem__(self, index):
        video = self.batch_frames[index][0]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)
        #N_frames:  (1, 100, 100, 3) -> frames, w, h, ch
        #N_masks:  (1, 100, 100) -> frames, w, h
        
        frame_idx = self.batch_frames[index][1]
        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame_idx))
        N_frames[0], coord = Crop_frames(Image.open(img_file).convert('RGB'), self.fixed_size)
        
        try:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame_idx))  
            N_masks[0], coord = Crop_frames(Image.open(mask_file).convert('P'), self.fixed_size, coord)
        except:
            N_masks[0] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        #Fs:  torch.Size([3, 3, 480, 854]) -> canais, frames, linhas, colunas
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(All_to_onehot(N_masks, self.K).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(All_to_onehot(N_masks, self.K).copy()).float()
            #Ms:  torch.Size([11, 3, 480, 854]) -> canais, frames, linhas, colunas
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info            
            #Chega pelo dataloader lá na chamada com uma dimensao extra:
            #Fs:  torch.Size([1, 3, 3, 480, 854])
            #Ms:  torch.Size([1, 11, 3, 480, 854])
            #num_objects:  tensor([[2]])
            #info:  {'name': ['bike-packing'], 'num_frames': tensor([69]), 
            #   'size_480p': [tensor([480]), tensor([854])]}
    
    def __len__(self):
        return len(self.batch_frames)
    
############################ DAVIS TEST ################################
class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling
    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB'))/255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                N_masks[f] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])            
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info
        

############################ AUX FUNCTIONS ################################


def Crop_frames(img, fixed_size, coord=None):
        
        fix_w, fix_h = fixed_size
        w, h = img.size[0], img.size[1]
        
        if coord is None:
            # resize
            if w <= h:
                #wsize = random.randrange(fix_w, max(fix_w+1,w))
                wsize = fix_w
                wpercent = (wsize/float(w))
                hsize = int((float(h)*float(wpercent)))                
            else:
                #hsize = random.randrange(fix_h, max(fix_h+1,h))
                hsize = fix_h
                wpercent = (hsize/float(h))
                wsize = int((float(w)*float(wpercent)))
            img = np.array(img.resize((hsize,wsize), Image.ANTIALIAS))/255.
                       
            # crop
            w, h = img.shape[0], img.shape[1]
            new_w = (random.randrange(0, w - fix_w) if w > fix_w else 0)
            new_h = (random.randrange(0, h - fix_h) if h > fix_h else 0)
            new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
             
        else:
            # resize
            wsize, hsize, new_w, new_h = coord
            img = np.array(img.resize((hsize,wsize), Image.ANTIALIAS), dtype=np.uint8)
            
            # crop
            w, h = img.shape[0], img.shape[1]
            new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
                   
        return new_img, (wsize, hsize, new_w, new_h)

def random_crop(frame_canvas, mask_canvas, frame, mask, wpad, hpad=None):
    
    if hpad is None:
        hpad = wpad
        
    w_paste, h_paste = frame.shape[0], frame.shape[1]
    w_canvas, h_canvas = frame_canvas.shape[0], frame_canvas.shape[1]
    
    w_paste_pos = random.randint(0, max(0, w_paste - w_canvas + wpad))
    h_paste_pos = random.randint(0, max(0, h_paste - h_canvas + hpad))
    
    w_paste_ini = max(0, w_paste_pos - wpad//2)
    h_paste_ini = max(0, h_paste_pos - hpad//2)
    
    w_paste_fin = min(w_paste, w_paste_pos + w_canvas - wpad//2)
    h_paste_fin = min(h_paste, h_paste_pos + h_canvas - hpad//2)
    
    w_canvas_pos_R = w_paste_fin - w_paste_pos
    h_canvas_pos_R = h_paste_fin - h_paste_pos
    
    w_canvas_pos_L = w_paste_pos - w_paste_ini
    h_canvas_pos_L = h_paste_pos - h_paste_ini
    
    w_canvas_ini = wpad//2 - w_canvas_pos_L
    h_canvas_ini = hpad//2 - h_canvas_pos_L
    
    w_canvas_fin = wpad//2 + w_canvas_pos_R
    h_canvas_fin = hpad//2 + h_canvas_pos_R
    
    wpi, wpf, hpi, hpf = w_paste_ini, w_paste_fin, h_paste_ini, h_paste_fin
    wci, wcf, hci, hcf = w_canvas_ini, w_canvas_fin, h_canvas_ini, h_canvas_fin
    
    frame_canvas[wci:wcf, hci:hcf] = frame[wpi:wpf, hpi:hpf]
    mask_canvas[wci:wcf, hci:hcf] = mask[wpi:wpf, hpi:hpf]
    
    return frame_canvas, mask_canvas

def resize_keeping_aspect_ratio(img, min_size=(384,384), coord=None):        
    img_w, img_h = img.size[0], img.size[1]
    (w_min,h_min) = min_size        
    
    if coord is None:
        if img_w <= img_h:
            wsize = random.randint(w_min, max(w_min, img_w))
            wpercent = (wsize/float(img_w))
            hsize = int((float(img_h)*float(wpercent)))                
        else:
            hsize = random.randint(h_min, max(h_min, img_h))
            wpercent = (hsize/float(img_h))
            wsize = int((float(img_w)*float(wpercent)))
        img = img.resize((wsize,hsize), Image.ANTIALIAS)
    else:
        wsize, hsize = coord
        img = img.resize((wsize,hsize), Image.ANTIALIAS)
    
    return np.asarray(img), (wsize, hsize)

def Skip_frames(frame, frame_skip, num_frames):
    
    if frame <= frame_skip:
        start_skip = 0
    else:
        start_skip = frame - frame_skip
        
    f1, f2 = sorted(random.sample(range(start_skip, frame), 2))
    
    return (f1, f2, frame)
        

if __name__ == '__main__':
    
    print('inicio')
    
    
    frame_skip = 5
    ##########
    # DAVIS trainset
    DATA_ROOT = '../rvos-master/databases'
    YEAR = 17
    SET = 'train'
    davis_Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p',
                                    imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16), frame_skip=frame_skip)
    
    # Youtube trainset
    youtube_Trainset = Youtube_MO_Train(DATA_ROOT, resolution='480p', 
                                        imset='train-train-meta.json', single_object=False, frame_skip=frame_skip)
    
    #concat DAVIS + Youtube
    trainset = data.ConcatDataset([davis_Trainset]+[youtube_Trainset])
    print('trainset instanciado, lenght: ', len(trainset))
    
    #train data loader
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    print('trainloader instanciado, lenght: ', len(trainloader))
    ##########
    
    dataiter = iter(trainloader)
    
    for cc in range(5):
        
        image, mask, n_obj, info = dataiter.next()
        
    
        N_frames = image[0].permute(1, 2, 3, 0)
        N_masks = mask[0].permute(1, 2, 3, 0)
        num_objects = n_obj[0].item()
            
        #print('N_frames: {}, {}'.format(N_frames.shape, N_frames.dtype))    
        #print('N_masks: {}, {}'.format(N_masks.shape, N_masks.dtype))    
        print('num_objects: {}'.format(num_objects))
        # N_frames: torch.Size([1, 3, 3, 384, 384]), torch.float32
        # N_masks: torch.Size([1, 11, 3, 384, 384]), torch.int32
        # num_objects: tensor([[1]]), torch.int64
        
        print('info valid samples', info['valid_samples'])
    
        for hh in range(num_objects+1):
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
    
    
    
    
