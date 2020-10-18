# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:37:55 2020

@author: Marcelo
"""
from config import cfg
import torch
import numpy as np
import sys
import argparse
import os
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
#import matplotlib.pyplot as plt
import torch.nn.functional as F
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import logging
import time
### My libs
from dataset import DAVIS_MO_Train, DAVIS_MO_Val, Youtube_MO_Train, Youtube_MO_Val
from dataset_pretrain import VOC_dataset, ECSSD_dataset, MSRA_dataset, SBD_dataset, COCO_dataset
from model import STM
from helpers import font, iou

# Constants
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="SST")    
    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch', help='epoch to load model',
                      default=-1, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default=MODEL_DIR, type=str)

    return parser.parse_args()

def get_dataloader(phase, batch_size, frame_skip):    
    if phase == 'main_train':
        datasets = cfg.MAIN_TRAIN_DATASETS
        trainset = []
        if 'davis' in datasets:   
            trainset += datasets['davis'] * [DAVIS_MO_Train(data_root=cfg.DATA_DAVIS, fixed_size=cfg.TRAIN_IMG_SIZE, frame_skip=frame_skip)]            
        if 'youtube' in datasets:
            trainset += datasets['youtube'] * [Youtube_MO_Train(data_root=cfg.DATA_YOUTUBE, fixed_size=cfg.TRAIN_IMG_SIZE, frame_skip=frame_skip)]
        return data.DataLoader(data.ConcatDataset(trainset), batch_size=batch_size, shuffle=True, num_workers=2)
        
    elif phase == 'main_val':
        datasets = cfg.MAIN_VAL_DATASETS
        valset = []
        if 'davis' in datasets:   
            valset += datasets['davis'] * [DAVIS_MO_Val(data_root=cfg.DATA_DAVIS, fixed_size=cfg.VAL_IMG_SIZE)]            
        if 'youtube' in datasets:
            valset += datasets['youtube'] * [Youtube_MO_Val(data_root=cfg.DATA_YOUTUBE, fixed_size=cfg.VAL_IMG_SIZE)]
        return data.DataLoader(data.ConcatDataset(valset), batch_size=batch_size, shuffle=False, num_workers=2)
    
    elif phase == 'pre_train':
        datasets = cfg.PRE_TRAIN_DATASETS
        pretrainset = []
        if 'voc' in datasets:
            pretrainset += datasets['voc'] * [VOC_dataset(data_root=cfg.DATA_VOC, fixed_size=cfg.TRAIN_IMG_SIZE, imset='train')]
        if 'ecssd' in datasets:
            pretrainset += datasets['ecssd'] * [ECSSD_dataset(data_root=cfg.DATA_ECSSD, fixed_size=cfg.TRAIN_IMG_SIZE, imset='train')]
        if 'msra' in datasets:
            pretrainset += datasets['msra'] * [MSRA_dataset(data_root=cfg.DATA_MSRA, fixed_size=cfg.TRAIN_IMG_SIZE, imset='train')]
        if 'sbd' in datasets:
            pretrainset += datasets['sbd'] * [SBD_dataset(data_root=cfg.DATA_SBD, fixed_size=cfg.TRAIN_IMG_SIZE, imset='train')]
        if 'coco' in datasets:
            pretrainset += datasets['coco'] * [COCO_dataset(data_root=cfg.DATA_COCO, fixed_size=cfg.TRAIN_IMG_SIZE, imset='train')]        
        return data.DataLoader(data.ConcatDataset(pretrainset), batch_size=batch_size, shuffle=True, num_workers=2)
    
    elif phase == 'pre_val':
        datasets = cfg.PRE_VAL_DATASETS
        prevalset = []
        if 'voc' in datasets:
            prevalset += datasets['voc'] * [VOC_dataset(data_root=cfg.DATA_VOC, fixed_size=cfg.VAL_IMG_SIZE, imset='val')]
        if 'ecssd' in datasets:
            prevalset += datasets['ecssd'] * [ECSSD_dataset(data_root=cfg.DATA_ECSSD, fixed_size=cfg.VAL_IMG_SIZE, imset='val')]
        if 'msra' in datasets:
            prevalset += datasets['msra'] * [MSRA_dataset(data_root=cfg.DATA_MSRA, fixed_size=cfg.VAL_IMG_SIZE, imset='val')]
        if 'sbd' in datasets:
            prevalset += datasets['sbd'] * [SBD_dataset(data_root=cfg.DATA_SBD, fixed_size=cfg.VAL_IMG_SIZE, imset='val')]
        if 'coco' in datasets:
            prevalset += datasets['coco'] * [COCO_dataset(data_root=cfg.DATA_COCO, fixed_size=cfg.VAL_IMG_SIZE, imset='val')]        
        return data.DataLoader(data.ConcatDataset(prevalset), batch_size=batch_size, shuffle=True, num_workers=2)

def run_train(train_phase='main_train', val_phase=None):    
    # get arguments
    args = get_arguments()
    
    if train_phase == 'main_train': 
        datasets= cfg.MAIN_TRAIN_DATASETS
        num_epochs = cfg.MAIN_TRAIN_EPOCHS
    else:
        datasets= cfg.PRE_TRAIN_DATASETS
        num_epochs = cfg.PRE_TRAIN_EPOCHS
    
    # Device infos
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
        print('Cuda version: ', torch.version.cuda)
        print('Current GPU id: ', torch.cuda.current_device())
        print('Device name: ', torch.cuda.get_device_name(device=torch.cuda.current_device()))
        print('Number of available devices:', num_devices)
    else:
        print('GPU is not available. CPU will be used.')
        device = torch.device("cpu")
        num_devices = 1
    
    # dataloader parameters
    train_batch_size = num_devices
    val_batch_size = 1
    frame_skip = 5
    
    # Isntantiate the model
    model = STM()
    print("STM model instantiated")    
    if torch.cuda.is_available():
        model = model.to(device)
        print("Model sent to cuda")
        if num_devices > 1:
            model = nn.DataParallel(model)
            print("Model parallelised in {} GPUs".format(num_devices))
    print('Using Datasets: ', datasets)
    
    # parameters, optmizer and loss
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            params += [{'params':[value]}]
    
    optimizer = torch.optim.Adam(params, lr=cfg.LR)
    criterion = F.cross_entropy
    
    writer = SummaryWriter()
    start_epoch = 0
    
    # load saved model if specified
    if args.loadepoch >= 0:
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))        
        # diretory name where models are saved
        load_name = os.path.join(args.output_dir,'{}.pth'.format(args.loadepoch)) #saved_models\#.pth  
        # get the state dict of current model 
        state = model.state_dict()
        # load entire saved model from checkpoint
        checkpoint = torch.load(load_name) # dict_keys(['epoch', 'model', 'optimizer'])
        # set next epoch to resume the training
        start_epoch = checkpoint['epoch'] + 1  
        # set frame_skip
        frame_skip = checkpoint['frame_skip']
        # filter out unnecessary keys from checkpoint
        checkpoint['model'] = {k:v for k,v in checkpoint['model'].items() if k in state}
        # overwrite entries in the existing state dict
        state.update(checkpoint['model'])
        # load the new state dict
        model.load_state_dict(state)
        # load optimizer state dict
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        print('  - complete!')       
    
    # instantiate dataloaders
    trainloader = get_dataloader(phase=train_phase, batch_size=train_batch_size, frame_skip=frame_skip)
    valloader = get_dataloader(phase=val_phase, batch_size=val_batch_size, frame_skip=0)
    iters_per_epoch = len(trainloader)
    print('len trainloader: ', iters_per_epoch)
    print('len valoader: ', len(valloader))
    
    # loop for training and validation
    for epoch in range(start_epoch, num_epochs):                
        
        # training
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
         
        # increases maximum frame skip by 5 (range 5->25) after 20 epochs
        if (epoch > 0) and (epoch % 20 == 0) and (train_phase == 'main_train'):
            frame_skip = min([frame_skip+5, 25])  
            
            # Update dataloader with new frame skip
            trainloader = get_dataloader(phase=train_phase, batch_size=train_batch_size, frame_skip=frame_skip)
        
        model.eval() #set eval mode to disable batchnorm and dropout
        running_loss = 0.0
        mean_iou = 0.0
          
        for seq, V in enumerate(trainloader):
            
            ############# interrupção só para testar
            if seq > 5:
                break
            
            Fs, Ms, num_objects, info = V
            
            if not info['valid_samples']:
                print('[TRAIN] sample idx: %d skiped' % (seq+1))
                continue
            
            Es = torch.zeros_like(Ms)
            #batch_size = 4:
            #Fs:  torch.Size([4, 3, 3, 384, 384])
            #Ms:  torch.Size([4, 11, 3, 384, 384])
            #num_objects:  tensor([[1],[1],[1],[1]], device='cuda:0')
            #Es:  torch.Size([4, 11, 3, 384, 384])
            
            # send input tensors to gpu
            if torch.cuda.is_available():
                Fs = Fs.to(device)
                Ms = Ms.to(device)
                num_objects = num_objects.to(device)
                Es = Es.to(device)
            
            loss = 0.0            
            Es[:,:,0] = Ms[:,:,0]
            keys, values = torch.Tensor([]), torch.Tensor([])
        
            #loop over the 3-1 frame+annotation samples (1st frame is used as reference)
            for t in range(1,3):
                
                # memorize
                prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects]))
                #prev_key(k4):  torch.Size([4, 11, 128, 1, 30, 54])
                #prev_value(v4):  torch.Size([4, 11, 512, 1, 30, 54])  
                
                if t-1 == 0:
                    this_keys, this_values = prev_key, prev_value # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)
                #t = 1:
                #this_keys:  torch.Size([4, 11, 128, 1, 30, 54])
                #this_values:  torch.Size([4, 11, 512, 1, 30, 54])
                
                # segment
                logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
                #logit: torch.Size([4, 11, 384, 384])                
                
                # compute probability maps for output
                Es[:,:,t] = F.softmax(logit, dim=1)
                #Es:  torch.Size([4, 11, 3, 384, 384])
                
                # update
                keys, values = this_keys, this_values
                
                # compute loss
                #loss += criterion(Es[:,:,t].clone(), Ms[:,:,t].float()) / train_batch_size  # replaced
                loss += criterion(logit, torch.argmax(Ms[:,:,t], dim=1))
                #logit: torch.Size([4, 11, 384, 384])
                #argmax(Ms[:,:,t], dim=1):  torch.Size([4, 384, 384])
                
            # backprop
            if loss > 0:  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            #compute intersection over union (iou)
            mean_iou += iou(Es[:,:,1:3], Ms[:,:,1:3])
            running_loss += loss.item()
            
            # logging and display
            #if (seq+1) % args.disp_interval == 0:
            if (seq+1) % cfg.DISP_INTERVAL == 0:                
                writer.add_scalar('Train/LOSS', running_loss/10, seq + epoch * iters_per_epoch)
                writer.add_scalar('Train/IOU', mean_iou/10, seq + epoch * iters_per_epoch)
                print('[TRAIN] idx: %d, loss: %.3f, iou: %.3f' % (seq+1, running_loss/10, mean_iou/10))
                running_loss = 0.0
                mean_iou = 0.0                   
            
        # saving checkpoint    
        if epoch % cfg.CHECKPOINT_EPOCHS == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch, 'frame_skip': frame_skip,'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),}, save_name)
            print('Model saved in: {}'.format(save_name))
            
        # validation
        if (val_phase is not None) and (epoch % cfg.EVAL_EPOCHS == 0) and (epoch > 0):
            with torch.no_grad():
                print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
                model.eval()
                run_validate(model, criterion, valloader, device, mem_every=5, mem_number=None)        
    

def run_validate(model, criterion, valloader, device, mem_every=None, mem_number=None):
        
    idx = 0
    next_change = 0
    
    for seq, V in enumerate(valloader):
                    
        ############# interrupção só para testar
        if seq > 5:
            break
        
        Fss, Mss, nums_objects, infos = V
        nums_frames = infos['num_frames']
        #Fss:  torch.Size([12, 3, 1, 384, 384]) -> val_batch_size = 12
        #Mss:  torch.Size([12, 11, 1, 384, 384]) -> val_batch_size = 12
        #nums_objects:  tensor([[1],[1],[1],[1]])
        #infos:  {'name': ['bear', 'bear', 'bear', 'bear'], 
        #       'num_frames': tensor([82, 82, 82, 82])}    
        
        if torch.cuda.is_available():
            Fss = Fss.to(device)
            Mss = Mss.to(device)
            nums_objects = nums_objects.to(device)
        
        batch_size = int(Fss.size(0)) 
        Es = torch.zeros(Mss[0,:,0].shape).unsqueeze(dim=0).to(device)
        #Es: torch.Size([1, 11, 384, 384])
                
        for batch_idx in range(batch_size):      
            
            Fs = Fss[batch_idx,:,0].unsqueeze(dim=0)
            Ms = Mss[batch_idx,:,0].unsqueeze(dim=0)
            #Fs:  torch.Size([1, 3, 384, 384])
            #Ms:  torch.Size([1, 11, 384, 384])
            num_objects = nums_objects[batch_idx]
            num_frames = nums_frames[batch_idx].item()
            
            # New sequence begins (fist frame)
            if idx == next_change:
                next_change += num_frames
                if mem_every:
                    to_memorize = [int(i) for i in np.arange(idx, next_change, step=mem_every)]
                elif mem_number:
                    to_memorize = [int(round(i)) for i in np.linspace(idx, next_change, num=mem_number+2)[:-1]]
                else:
                    raise NotImplementedError
                #If mem_every = 5, then to_memorize = [0, 5, 10, 15...]
                
                keys, values = torch.Tensor([]), torch.Tensor([])
                Es = Ms
                loss = 0.0       
                mean_iou = 0.0
                first_frame = True
            else:
                first_frame = False
        
            # memorize
            with torch.no_grad():
                prev_key, prev_value = model(Fs, Es, torch.tensor([num_objects]))
                #prev_key(k4):  torch.Size([1, 11, 128, 1, 30, 57])
                #prev_value(v4):  torch.Size([1, 11, 512, 1, 30, 57])
     
            if first_frame: # 
                this_keys, this_values = prev_key, prev_value # only prev memory
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)
                #this_keys:  torch.Size([1, 11, 128, 1, 7, 7])
                #this_values:  torch.Size([1, 11, 512, 1, 7, 7])

            # segment
            with torch.no_grad():
                
                if not first_frame:
                    logit = model(Fs, this_keys, this_values, torch.tensor([num_objects]))
                    #logit:  torch.Size([1, 11, 384, 384])            
               
                    Es = F.softmax(logit, dim=1)
                    #Es: torch.Size([1, 11, 384, 384])
                    
                    # compute loss
                    loss += criterion(logit, torch.argmax(Ms, dim=1)).item()
                    #torch.argmax(Ms, dim=1):  torch.Size([1, 384, 384])
                    
                    #compute intersection over union (iou)
                    mean_iou += iou(Es.unsqueeze(2), Ms.unsqueeze(2))
                
                # update
                if first_frame or idx in to_memorize:
                    keys, values = this_keys, this_values
            
            if idx == next_change - 1: #last frame of the sequence
                print("[VAL] idx: %d, loss: %.3f, iou: %.3f" % (idx+1, loss/(num_frames-1), mean_iou/(num_frames-1)))  
                
            idx += 1              

        
if __name__ == "__main__":    
    
    print(">>>>\nMy STM starting...")    
    print('Python version: ', sys.version)   
    print('Pytorch version: ', torch.__version__)    
    run_train(train_phase='main_train', val_phase='main_val')
    #run_train(train_phase='pre_train', val_phase='pre_val')
    
    
    
           