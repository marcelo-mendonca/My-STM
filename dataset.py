
from config import cfg
import os.path
import numpy as np
from PIL import Image
import torch
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
from dataset_pretrain import All_to_onehot, bbox2, validate_samples


############################ YOUTUBE TRAIN ################################
class Youtube_MO_Train(data.Dataset):    
    def __init__(self, data_root, imset='train', resolution='480p', single_object=False, frame_skip=5, fixed_size=(384,384)):
        #../rvos-master/databases/YouTubeVOS/train
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
        
        mask_dir = os.path.join(data_root, 'train', 'Annotations')
        image_dir = os.path.join(data_root, 'train', 'JPEGImages')
        _imset_dir = os.path.join(data_root, 'train')
        _imset = 'train-train-meta.json' if imset == 'train' else 'train-val-meta.json'
        _imset_f = os.path.join(_imset_dir, _imset)
        
        self.videos = []
        self.num_frames = {}
        self.frame_paths = {}
        self.mask_paths = {}
        self.num_objects = {}
        self.train_triplets = {}
        self.frame_skip = frame_skip
        self.fixed_size = fixed_size
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
        
        for idx, f in enumerate(train_triplet):
            
            frame_path = self.frame_paths[video][f]
            frame, new_size = random_resize(Image.open(frame_path).convert('RGB'), self.fixed_size)
            
            try:
                mask_path = self.mask_paths[video][f]
                mask, _ = random_resize(Image.open(mask_path).convert('P'), self.fixed_size, new_size)
            except:
                mask = np.zeros(new_size, dtype=np.uint8)
            #masks:  numpy (w>=fixed_size, h>=fixed_size)
            
            frame_transf, mask_transf = self.affine_transf(self.frame_transf(frame.copy()), self.mask_transf(mask.copy()))
            #frame_transf:  torch (c, w, h) -> c=3
            #mask_transf:  torch (c, w, h) -> c=1
            
            new_frame = ChannelsLast()(frame_transf)
            np_mask = modal(np.array(mask_transf[0]).astype(np.uint8), disk(5)) # modal function clean up some noise caused by affine_transf
            new_mask = torch.from_numpy(np_mask) 
            #new_frame:  torch (w, h, c)
            #new_mask:  torch (w, h)
            
            try:
                bbox = bbox2(np_mask)
            except:
                bbox = None
            N_frames[idx], coord = random_crop(new_frame, self.fixed_size, bbox)            
            N_masks[idx], _ = random_crop(new_mask, self.fixed_size, bbox, coord)            
            #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
            #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
        
        info['valid_samples'] = validate_samples(N_masks, self.num_objects[video])      
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
    def __init__(self, data_root, imset='train', resolution='480p', single_object=False, frame_skip=5, fixed_size=(384,384)):
        #../rvos-master/databases/DAVIS2017
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
        
        self.mask_dir = os.path.join(data_root, 'Annotations', resolution)
        #self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(data_root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(data_root, 'ImageSets')
        _imset = '2017/train.txt' if imset == 'train' else '2017/val.txt'
        _imset_f = os.path.join(_imset_dir, _imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.train_triplets = {}
        self.frame_skip = frame_skip
        self.fixed_size = fixed_size
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
                self.num_objects[_video] = np.max(np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P")))
                
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
        
        for idx, f in enumerate(train_triplet):
            
            frame_path = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            frame, new_size = random_resize(Image.open(frame_path).convert('RGB'), self.fixed_size)
            
            try:
                mask_path = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f)) 
                mask, _ = random_resize(Image.open(mask_path).convert('P'), self.fixed_size, new_size)
            except:
                mask = np.zeros(new_size, dtype=np.uint8)
            #masks:  numpy (w>=fixed_size, h>=fixed_size)
                
            frame_transf, mask_transf = self.affine_transf(self.frame_transf(frame.copy()), self.mask_transf(mask.copy()))
            #frame_transf:  torch (c, w, h) -> c=3
            #mask_transf:  torch (c, w, h) -> c=1
            
            new_frame = ChannelsLast()(frame_transf)
            np_mask = modal(np.array(mask_transf[0]).astype(np.uint8), disk(5)) # modal function clean up some noise caused by affine_transf
            new_mask = torch.from_numpy(np_mask) 
            #new_frame:  torch (w, h, c)
            #new_mask:  torch (w, h)
            
            try:
                bbox = bbox2(np_mask)
            except:
                bbox = None
            N_frames[idx], coord = random_crop(new_frame, self.fixed_size, bbox)            
            N_masks[idx], _ = random_crop(new_mask, self.fixed_size, bbox, coord)
            #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
            #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
        
        info['valid_samples'] = validate_samples(N_masks, self.num_objects[video])
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
    def __init__(self, data_root, imset='val', resolution='480p', single_object=False, fixed_size=(384,384)):
        #../rvos-master/databases/YouTubeVOS/train
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
        
        mask_dir = os.path.join(data_root, 'train', 'Annotations')
        image_dir = os.path.join(data_root, 'train', 'JPEGImages')
        _imset_dir = os.path.join(data_root, 'train')
        _imset = 'train-val-meta.json' if imset == 'val' else 'train-train-meta.json' 
        _imset_f = os.path.join(_imset_dir, _imset)
        
        self.videos = []
        self.num_frames = {}
        self.frame_paths = {}
        self.mask_paths = {}
        self.num_objects = {}
        self.batch_frames = {}
        self.fixed_size = fixed_size
        self.frame_transf = Compose([ToTensor(), TypeCast('float'), RangeNormalize(0, 1)])
        self.mask_transf = Compose([ToTensor(), TypeCast('float')])
        idx = 0 
        
        with open(_imset_f) as json_file:
            json_data = edict(json.load(json_file))
            for _video in json_data.videos.keys():
                self.videos.append(_video)
                self.frame_paths[_video] = glob.glob(os.path.join(image_dir, _video, '*.jpg'))
                self.num_frames[_video] = len(self.frame_paths[_video])
                self.mask_paths[_video] = glob.glob(os.path.join(mask_dir, _video, '*.png'))
                self.num_objects[_video] = len(json_data.videos[_video].objects)
                
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
        info['valid_samples'] = True
        
        N_frames = torch.zeros(1, self.fixed_size[0], self.fixed_size[1], 3)
        N_masks = torch.zeros(1, self.fixed_size[0], self.fixed_size[1]) 
        #N_frames:  (n, w, h, c) -> num frames, widith, height, channel
        #N_masks:  (n, w, h)       
        
        frame_idx = self.batch_frames[index][1]
        frame_path = self.frame_paths[video][frame_idx]
        frame, new_size = random_resize(Image.open(frame_path).convert('RGB'), self.fixed_size)
        #frame:  numpy (w>=fixed_size[0], h>=fixed_size[1], c)
        
        try:
            mask_path = self.mask_paths[video][frame_idx]
            mask, _ = random_resize(Image.open(mask_path).convert('P'), self.fixed_size, new_size)
        except:
            mask = np.zeros(new_size, dtype=np.uint8)
        #masks:  numpy (w>=fixed_size, h>=fixed_size)
            
        new_frame = self.frame_transf(frame.copy())
        new_mask = self.mask_transf(mask.copy())
        
        try:
            bbox = bbox2(mask)
        except:
            bbox = None
        N_frames[0], coord = random_crop(new_frame, self.fixed_size, bbox)            
        N_masks[0], _ = random_crop(new_mask, self.fixed_size, bbox, coord)            
        #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
        #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
        
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
        return len(self.batch_frames)
    
############################ DAVIS VALIDATION ################################
class DAVIS_MO_Val(data.Dataset):
    def __init__(self, data_root, imset='val', resolution='480p', single_object=False, fixed_size=(384,384)):
        #../rvos-master/databases/DAVIS2017
        if not os.path.isdir(data_root):
            raise RuntimeError('Dataset not found or corrupted: {}'.format(data_root))
        
        self.mask_dir = os.path.join(data_root, 'Annotations', resolution)
        self.image_dir = os.path.join(data_root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(data_root, 'ImageSets')
        _imset = '2017/val.txt' if imset == 'val' else '2017/train.txt'
        _imset_f = os.path.join(_imset_dir, _imset)
        
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.batch_frames = {}
        self.fixed_size = fixed_size
        self.frame_transf = Compose([ToTensor(), TypeCast('float'), RangeNormalize(0, 1)])
        self.mask_transf = Compose([ToTensor(), TypeCast('float')])
        idx = 0
        
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                
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
        info['valid_samples'] = True
        
        N_frames = torch.zeros(1, self.fixed_size[0], self.fixed_size[1], 3)
        N_masks = torch.zeros(1, self.fixed_size[0], self.fixed_size[1]) 
        #N_frames:  (n, w, h, c) -> num frames, widith, height, channel
        #N_masks:  (n, w, h)       
        
        frame_idx = self.batch_frames[index][1]
        frame_path = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(frame_idx))
        frame, new_size = random_resize(Image.open(frame_path).convert('RGB'), self.fixed_size)
        #frame:  numpy (w>=fixed_size[0], h>=fixed_size[1], c)
        
        try:
            mask_path = os.path.join(self.mask_dir, video, '{:05d}.png'.format(frame_idx))
            mask, _ = random_resize(Image.open(mask_path).convert('P'), self.fixed_size, new_size)
        except:
            mask = np.zeros(new_size, dtype=np.uint8)
        #masks:  numpy (w>=fixed_size, h>=fixed_size)
            
        new_frame = self.frame_transf(frame.copy())
        new_mask = self.mask_transf(mask.copy())
        
        try:
            bbox = bbox2(mask)
        except:
            bbox = None
        N_frames[0], coord = random_crop(new_frame, self.fixed_size, bbox)            
        N_masks[0], _ = random_crop(new_mask, self.fixed_size, bbox, coord)            
        #N_frames: torch (n, w=fixed_size[0], h=fixed_size[1], c) -> torch.float32
        #N_masks: torch (n, w=fixed_size[0], h=fixed_size[1]) -> torch.float32
        
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
            
def random_resize(img, min_size, new_size=None):    
    min_w, min_h = min_size
    w, h = img.size[0], img.size[1]
    
    if new_size is None:
        if w <= h:
            wsize = random.randrange(min_w, max(min_w+1,w))
            wpercent = (wsize/float(w))
            hsize = int((float(h)*float(wpercent)))                
        else:
            hsize = random.randrange(min_h, max(min_h+1,h))
            wpercent = (hsize/float(h))
            wsize = int((float(w)*float(wpercent)))
        new_img = np.array(img.resize((wsize,hsize), Image.ANTIALIAS))/255.         
    else:
        wsize, hsize = new_size
        new_img = np.array(img.resize((wsize,hsize), Image.ANTIALIAS), dtype=np.uint8)
               
    return new_img, (wsize, hsize)

def random_crop(img, fixed_size, bbox=None, coord=None):    
    fix_w, fix_h = fixed_size
    w, h = img.shape[0], img.shape[1]
    
    if coord is None:
        if bbox is not None:
            rmin, rmax, cmin, cmax = bbox
        else:
            rmin, rmax, cmin, cmax = 0, w, 0, h
            
        pad_w = np.floor((rmax-rmin)/10)
        pad_h = np.floor((cmax-cmin)/10)
        
        w_L = min(rmin, rmax - pad_w - fix_w)
        w_R = max(rmin + pad_w, rmax - fix_w)
        
        h_U = min(cmin, cmax - pad_h - fix_h)
        h_D = max(cmin + pad_h, cmax - fix_h)
        
        try:
            pos_w = random.randrange(max(0, w_L), min(w_R, w - fix_w))
        except:
            pos_w = 0
        try:
            pos_h = random.randrange(max(0, h_U), min(h_D, h - fix_h))
        except:
            pos_h = 0
        
        new_img = img[pos_w:pos_w+fix_w, pos_h:pos_h+fix_h]    
    
    else:
        pos_w, pos_h = coord
        new_img = img[pos_w:pos_w+fix_w, pos_h:pos_h+fix_h]
    
    return new_img, (pos_w, pos_h)


def Skip_frames(frame, frame_skip, num_frames):    
    if frame <= frame_skip:
        start_skip = 0
    else:
        start_skip = frame - frame_skip
        
    f1, f2 = sorted(random.sample(range(start_skip, frame), 2))
    
    return (f1, f2, frame)

############################ MAIN ################################
if __name__ == '__main__':
    
    print('inicio')
    
    import time
    FIXED_SIZE = (150, 150)
    frame_skip = 5
    ##########
    

    # TRAIN
    # Davis trainset
    davis_Trainset = DAVIS_MO_Train(data_root=cfg.DATA_DAVIS, frame_skip=frame_skip, fixed_size=FIXED_SIZE)    
    # Youtube trainset
    youtube_Trainset = Youtube_MO_Train(data_root=cfg.DATA_YOUTUBE, frame_skip=frame_skip, fixed_size=FIXED_SIZE)    
    #concat DAVIS + Youtube
    trainset = data.ConcatDataset(5*[davis_Trainset]+[youtube_Trainset])    
    #train data loader
    trainloader = data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    
    print('trainloader lenght: ', len(trainloader))
    
    # # VALIDATION
    # # Davis valset
    # davis_Valset = DAVIS_MO_Val(data_root=cfg.DATA_DAVIS, fixed_size=FIXED_SIZE)    
    # # Youtube valset
    # youtube_Valset = Youtube_MO_Val(data_root=cfg.DATA_YOUTUBE, fixed_size=FIXED_SIZE)    
    # #concat DAVIS + Youtube
    # valset = data.ConcatDataset(5*[davis_Valset]+[youtube_Valset])    
    # #train data loader
    # valloader = data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=0)
    
    # print('valloader lenght: ', len(valloader))
    
    # input('Press Enter button to continue...')
    
    # trainloader lenght:  91777
    # valloader lenght:  27607
    ##########
    t0= time.process_time()
    dataiter = iter(trainloader)
    
    validos = 0
    invalidos = 0
    for cc in range(200):
        
        _, _, _, info = dataiter.next()
        
        t1 = time.process_time() - t0
        print("{}) = {}, Time: {:.3f}".format(cc, info['valid_samples'], t1)) # CPU seconds elapsed (floating point)
        t0 = t1
        
        if info['valid_samples']:
            validos +=1
        else:
            invalidos +=1
    
    print('boas {}, ruins {}'.format(validos, invalidos))
        
    
        #N_frames = image[0].permute(1, 2, 3, 0)
        #N_masks = mask[0].permute(1, 2, 3, 0)
        #num_objects = n_obj[0].item()
            
        #print('N_frames: {}, {}'.format(N_frames.shape, N_frames.dtype))    
        #print('N_masks: {}, {}'.format(N_masks.shape, N_masks.dtype))    
        #print('num_objects: {}'.format(num_objects))
        # N_frames: torch.Size([1, 3, 3, 384, 384]), torch.float32
        # N_masks: torch.Size([1, 11, 3, 384, 384]), torch.int32
        # num_objects: tensor([[1]]), torch.int64
        
        
        
        # if info['valid_samples'][0] and (info['valid_samples'][1] or info['valid_samples'][2]):
        #     validos += 1
        #     resu = 'valido'
        # else:
        #     invalidos += 1
        #     resu = 'invalido'
        
        # print('sample ({}/10) {} = {}'.format(cc, info['valid_samples'], resu))
    
    # print('\n\n')
    # print('validos: {}/1000'.format(validos))
    # print('invalidos: {}/1000'.format(invalidos))
    
    #1) validos: 737/1000
    #1) invalidos: 263/1000
    
    #2) validos: 865/1000
    #2) invalidos: 135/1000
    
    #3) validos: 984/1000
    #3) invalidos: 16/1000
    
    #4) validos: 982/1000
    #4) invalidos: 18/1000
    
    #5) validos: 982/1000
    #5) invalidos: 18/1000
    
        # TRAIN
        # for hh in range(1):
        #     print('Mask CH: ', hh)
        #     ff = plt.figure()
        #     ff.add_subplot(2,3,1)
        #     plt.imshow(N_frames[0])
        #     ff.add_subplot(2,3,2)
        #     plt.imshow(N_frames[1])
        #     ff.add_subplot(2,3,3)
        #     plt.imshow(N_frames[2])
        #     ff.add_subplot(2,3,4)
        #     plt.imshow(N_masks[0,:,:,hh])
        #     ff.add_subplot(2,3,5)
        #     plt.imshow(N_masks[1,:,:,hh])
        #     ff.add_subplot(2,3,6)
        #     plt.imshow(N_masks[2,:,:,hh])
        #     plt.show(block=True) 
            
        #     input("Press Enter to continue...") 
        
        # VALIDATION
        # for hh in range(1):
        #     print('Mask CH: ', hh)
        #     ff = plt.figure()
        #     ff.add_subplot(1,2,1)
        #     plt.imshow(N_frames[0])
        #     ff.add_subplot(1,2,2)
        #     plt.imshow(N_masks[0,:,:,hh])
        #     plt.show(block=True) 
            
        #     input("Press Enter to continue...") 
    
    
    
    
