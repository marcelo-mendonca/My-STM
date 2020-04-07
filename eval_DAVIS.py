from __future__ import division
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
import math
import time
import tqdm
import os
import argparse
import copy
import sys


### My libs
from dataset import DAVIS_MO_Test
from model import STM

def main():
    print(">>>> STM starting <<<<")
    print ("Python version: ", sys.version)
    print("Pytorch version: ", torch.__version__)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('Using {} cuda device(s)'.format(torch.cuda.device_count()))
        print("Device ", device)
        print("Cuda version: ", torch.version.cuda)
        run_test(device)
    else:
        print('cuda is not available...')

def get_arguments():
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    return parser.parse_args()

def Run_video(model, Fs, Ms, num_frames, num_objects, Mem_every=None, Mem_number=None):
    #model = STM()
    #Fs:  torch.Size([1, 3, 69, 480, 910])
    #Ms:  torch.Size([1, 11, 69, 480, 910])
    #num_frames: 69
    #num_objects:  2 
    
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError
    #Se mem_every=5, então to_memorize = [0, 5, 10, 15...]
    

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]
    #Es:  torch.Size([1, 11, 69, 480, 910])

    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects]))
            #prev_key(k4):  torch.Size([1, 11, 128, 1, 30, 57])
            #prev_value(v4):  torch.Size([1, 11, 512, 1, 30, 57])

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        #(t=1) this_keys: torch.Size([1, 11, 128, 1, 30, 57])
        #(t=1) this_values: torch.Size([1, 11, 512, 1, 30, 57])
        #(t=2) this_keys: torch.Size([1, 11, 128, 2, 30, 57])
        #(t=2) this_values: torch.Size([1, 11, 512, 2, 30, 57])
        #(t=3, 4, 5, 6): não muda (mem_every = 5)
        #(t=7) this_keys: torch.Size([1, 11, 128, 3, 30, 57])
        #(t=7) this_values: torch.Size([1, 11, 512, 3, 30, 57])
        
        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
            #(t=39) logit: torch.Size([1, 11, 480, 910])
        
        Es[:,:,t] = F.softmax(logit, dim=1)
        #(t=39) Es: torch.Size([1, 11, 69, 480, 910])

        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    #(t=39) pred: (69, 480, 910)
    return pred, Es

def run_test(cuda_device):
    torch.set_grad_enabled(False) # Volatile
       
    args = get_arguments()
    
    GPU = args.g
    YEAR = args.y
    SET = args.s
    VIZ = args.viz
    DATA_ROOT = args.D
    device = cuda_device
    
    # Model and version
    MODEL = 'STM'
    print(MODEL, ': Testing on DAVIS', YEAR)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    print('--- CUDA:')
    
    
    if VIZ:
        print('--- Produce mask overaid video outputs. Evaluation will run slow.')
        print('--- Require FFMPEG for encoding, Check folder ./viz')
    
    
    palette = Image.open(DATA_ROOT + '/Annotations/480p/blackswan/00000.png').getpalette()
    
    
    Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    
    model = nn.DataParallel(STM())
    
    try:
        model = model.to(device)
        print('--- Model successfully sent to cuda')
    except:
        print('--- Cuda is not available')
       
    model.eval() # turn-off BN
    
    pth_path = 'STM_weights.pth'
    print('Loading weights:', pth_path)
    model.load_state_dict(torch.load(pth_path))
    
    code_name = '{}_DAVIS_{}_{}'.format(MODEL,YEAR,SET)
    print('Start Testing:', code_name)    
    
    for seq, V in enumerate(Testloader):
#        if seq < 21:
#            continue
        Fs, Ms, num_objects, info = V
        #Fs:  torch.Size([1, 3, 69, 480, 910])
        #Ms:  torch.Size([1, 11, 69, 480, 910])
        #num_objects:  tensor([[2]])
        #info:  {'name': ['bike-packing'], 'num_frames': tensor([69]), 
        #   'size_480p': [tensor([480]), tensor([910])]}
        
        seq_name = info['name'][0] # = bike-packing
        num_frames = info['num_frames'][0].item() # = 69
        if num_frames > 40:
            num_frames = 40
        print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
        
        pred, Es = Run_video(model, Fs, Ms, num_frames, num_objects, Mem_every=5, Mem_number=None)
        #pred:  (69, 480, 910)
        #Es:  torch.Size([1, 11, 69, 480, 910])
            
        # Save results for quantitative eval ######################
        test_path = os.path.join('./test', code_name, seq_name)
        print('test_path: ', test_path)
        
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        for f in range(num_frames):
            img_E = Image.fromarray(pred[f])
            #img_E:  (910, 480)
            #imagem com pixels rotulados de 0 a Nº de objetos
            img_E.putpalette(palette)
            #putpalette troca os rótulos pelos mesmos da anotação manual de
            #onde se extraiu o palette
            img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
        
        #Código para visualizar a máscara de segmentação sobre o frame:
        if VIZ:
            from helpers import overlay_davis
            # visualize results #######################
            viz_path = os.path.join('./viz/', code_name, seq_name)
            if not os.path.exists(viz_path):
                os.makedirs(viz_path)
    
            for f in range(num_frames):
                pF = (Fs[0,:,f].permute(1,2,0).numpy() * 255.).astype(np.uint8)
                pE = pred[f]
                canvas = overlay_davis(pF, pE, palette)
                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(viz_path, 'f{}.jpg'.format(f)))
    
            #vid_path = os.path.join('./viz/', code_name, '{}.mp4'.format(seq_name))
            #frame_path = os.path.join('./viz/', code_name, seq_name, 'f%d.jpg')
            #os.system('ffmpeg -framerate 10 -i {} {} -vcodec libx264 -crf 10  -pix_fmt yuv420p  -nostats -loglevel 0 -y'.format(frame_path, vid_path))
    
if __name__ == "__main__":
    
    main()

#Garbage
#plt.matshow(B_['o'].permute(1,2,0)[:,:,0])
#plt.show()
#input("Press Enter to continue...")