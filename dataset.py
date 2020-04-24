import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob
import random

class DAVIS_MO_Train(data.Dataset):
    
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
        self.train_triplets = {}
        self.frame_skip = 5
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
                
                #starts from 2 so that there are at least 2 frames before
                for f in range(2, self.num_frames[_video]):
                    self.train_triplets[idx] = (_video, f)
                    idx += 1
        self.K = 11
        self.single_object = single_object

    def __getitem__(self, index):
        video = self.train_triplets[index][0]
        #video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((3,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+self.shape[video], dtype=np.uint8)
        #N_frames:  (3, 100, 100, 3)
        #N_masks:  (3, 100, 100)
        
        train_triplet = self.Skip_frames(self.train_triplets[index][1], self.num_frames[video])
        #train_triplet = (3, 5, 9) -> frames: t=3, t=5, t=9
        
        #for f in range(self.num_frames[video]):
        for idx, f in enumerate(train_triplet):
            #print('index: {}, triplet: {}, name: {}'.format(index, train_triplet, video))
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[idx], coord = self.crop_frames(np.array(Image.open(img_file).convert('RGB'))/255.)
            #sem crop: N_frames[idx]:  (480, 854, 3)
            #com crop: N_frames[idx]:  (384, 384, 3)
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[idx], coord = self.crop_frames(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), coord)
            except:
                # print('a')
                N_masks[idx] = 255
            #sem crop: N_masks[idx]:  (480, 854)
            #com crop: N_masks[idx]:  (384, 384)
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        #Fs:  torch.Size([3, 3, 480, 854]) -> canais, frames, linhas, colunas
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
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
        return len(self.train_triplets)    
    
    def crop_frames(self, img, coord=None):
        
        fix_w, fix_h = self.fixed_size
        
        if coord:
            new_w, new_h = coord
            if len(img.shape) == 3:
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h,:]
            else:
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
        else:
            if len(img.shape) == 3:
                w, h, _ = img.shape
                new_w = random.randrange(0, w - fix_w)
                new_h = random.randrange(0, h - fix_h)
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h,:]
            else:
                w, h = img.shape
                new_w = random.randrange(0, w - fix_w)
                new_h = random.randrange(0, h - fix_h)
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
        
        coord = (new_w, new_h)
        
        return new_img, coord
    
    def Skip_frames(self, frame, num_frames):
        
        if frame <= self.frame_skip:
            start_skip = 0
        else:
            start_skip = frame - self.frame_skip
            
        f1, f2 = sorted(random.sample(range(start_skip, frame), 2))
        
        return (f1, f2, frame)
    
    def Set_frame_skip(self, epoch):
        #frame_skip is increased by 5 at every 20 epoch during main-training
        if (epoch > 0) and (epoch % 20 == 0):
            self.frame_skip = min([self.frame_skip+5, 25])
    
    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        #M:  (11, 480, 854)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        #num_objects:  2
        #masks:  (3, 480, 854) -> 3 máscaras do train_triplet
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]): #n -> 0, 1, 2
            Ms[:,n] = self.To_onehot(masks[n])
        #Ms:  (11, 3, 480, 854)
        return Ms


class DAVIS_MO_Val(data.Dataset):
    
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
        self.batch_frames = {}
        self.frame_skip = 5
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
        #video = self.videos[index]
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
        N_frames[0], coord = self.crop_frames(np.array(Image.open(img_file).convert('RGB'))/255.)
        
        try:
            mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame_idx))  
            N_masks[0], coord = self.crop_frames(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8), coord)
        except:
            N_masks[0] = 255
        
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        #Fs:  torch.Size([3, 3, 480, 854]) -> canais, frames, linhas, colunas
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
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
    
    def crop_frames(self, img, coord=None):
        
        fix_w, fix_h = self.fixed_size
        
        if coord:
            new_w, new_h = coord
            if len(img.shape) == 3:
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h,:]
            else:
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
        else:
            if len(img.shape) == 3:
                w, h, _ = img.shape
                new_w = random.randrange(0, w - fix_w)
                new_h = random.randrange(0, h - fix_h)
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h,:]
            else:
                w, h = img.shape
                new_w = random.randrange(0, w - fix_w)
                new_h = random.randrange(0, h - fix_h)
                new_img = img[new_w:new_w+fix_w, new_h:new_h+fix_h]
        
        coord = (new_w, new_h)
        
        return new_img, coord
    
    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        #M:  (11, 480, 854)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        #num_objects:  2
        #masks:  (3, 480, 854) -> 3 máscaras do train_triplet
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]): #n -> 0, 1, 2
            Ms[:,n] = self.To_onehot(masks[n])
        #Ms:  (11, 3, 480, 854)
        return Ms


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


if __name__ == '__main__':
    pass
