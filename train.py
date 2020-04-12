# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:37:55 2020

@author: Marcelo
"""
import torch
import sys
import argparse
import os
from torch.utils import data
import torch.nn as nn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter

### My libs
from dataset import DAVIS_MO_Test, DAVIS_MO_Train
from model import STM
from helpers import *

# Constants
MODEL_DIR = 'saved_models'
NUM_EPOCHS = 1000


def main():
    print(">>>> Hello world! \nNew STM training script starting...")
    
    print('Python version: ', sys.version)   
    print('Pytorch version: ', torch.__version__)    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device ', device)
    print('Cuda version: ', torch.version.cuda)
    if torch.cuda.is_available():
        print('Number of devices available:', torch.cuda.device_count())
        run_train(device)
    else:
        print('Cuda is not available...')


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="SST")
    parser.add_argument("-g", type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument("-s", type=str, help="set", required=True)
    parser.add_argument("-y", type=int, help="year", required=True)
    parser.add_argument("-viz", help="Save visualization", action="store_true")
    parser.add_argument("-D", type=str, help="path to data",default='/local/DATA')
    #new ones:
    
    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch', help='epoch to load model',
                      default=-1, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default=MODEL_DIR, type=str)
    # config
    parser.add_argument('--epochs', dest='num_epochs',
                      help='number of epochs to train',
                      default=NUM_EPOCHS, type=int)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-5, type=float)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='interval of epochs to perform validation',
                      default=10, type=int)
    
    #return
    return parser.parse_args()


def run_train(device):
    
    # get arguments
    args = get_arguments()
    
    GPU = args.g
    YEAR = args.y
    SET = args.s
    VIZ = args.viz
    DATA_ROOT = args.D
    
    # data loader
    Trainset = DAVIS_MO_Train(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    #Trainset = DAVIS(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), multi_object=(YEAR==17))
    Trainloader = data.DataLoader(Trainset, batch_size=1, shuffle=False, num_workers=1)
    
    Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)
    
    # Intantiate the model
    model = nn.DataParallel(STM())
    
    if torch.cuda.is_available():
        model.cuda()
    
    writer = SummaryWriter()
    start_epoch = 0
    
    # load saved model if specified
    if args.loadepoch >= 0:
        print('Loading checkpoint {}@Epoch {}{}...'.format(font.BOLD, args.loadepoch, font.END))
        load_name = os.path.join(args.output_dir,
          '{}.pth'.format(args.loadepoch))
        state = model.state_dict()
        checkpoint = torch.load(load_name)
        start_epoch = checkpoint['epoch'] + 1
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        model.load_state_dict(state)
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        del checkpoint
        torch.cuda.empty_cache()
        print('  - complete!')
    
    
    # params
    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        #params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]
        params += [{'params':[value],'lr':args.lr}]
    print('Parameters lenght: ', len(params))
    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params, lr=args.lr)
    iters_per_epoch = len(Trainloader)
    
    # loop for training/validation
    for epoch in range(start_epoch, args.num_epochs):
        
        # testing
        if epoch % args.eval_epoch == 1:
            print("Testing...")
            with torch.no_grad():
                print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
                model.eval()
                loss = 0
                iOU = 0
                print("Testing routine still incomplete...")
        
        # training
        model.eval() #set eval mode to disable batchnorm and dropout
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        
        #maximum_skip is increased by 5 at every 20 epoch during main-training
        #Trainset.frame_skip initial value is 5
        if (epoch > 0) and (epoch % 20 == 0):
            Trainset.frame_skip = min([Trainset.frame_skip+5, 25])
          
        for seq, V in enumerate(Trainloader):
            
            Fs, Ms, num_objects, info = V
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0].item()
            
            ############################
            #ATENÇÃO: deu ruim qd coloca batch > 1 por causa de num_objects
            #tem q ver como resolver, especialmente em model -> Pad_mem que
            #tá dando erro nessa merda...                    
            
            optimizer.zero_grad()
            
            tt = time.time()
            
            Es = torch.zeros_like(Ms)
            Es[:,:,0] = Ms[:,:,0]
            #Es recebe de Ms a máscara com menor indice temporal; as outras posições de Es ficam vazias
            #Fs:  torch.Size([1, 3, 3, 480, 854])
            #Ms:  torch.Size([1, 11, 3, 480, 854])
            #Es:  torch.Size([1, 11, 3, 480, 854])
            #Fs:  torch.Size([1, 3, 3, 384, 384])
            #Ms:  torch.Size([1, 11, 3, 384, 384])
            #num_objects:  torch.Size([1, 1])
            
            loss = 0
            
            #loop over the 3 frame+annotation samples
            for t in range(1,3):
                
                # memorize torch.tensor([num_objects])
                prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects]))
                #prev_key(k4):  torch.Size([1, 11, 128, 1, 30, 54])
                #prev_value(v4):  torch.Size([1, 11, 512, 1, 30, 54])  
                
                if t-1 == 0:
                    this_keys, this_values = prev_key, prev_value # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)
                #t = 1:
                #this_keys:  torch.Size([1, 11, 128, 1, 30, 54])
                #this_values:  torch.Size([1, 11, 512, 1, 30, 54])
                
                # segment
                logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
                #(t=39) logit: torch.Size([1, 11, 480, 910])
                
                
                Es[:,:,t] = F.softmax(logit, dim=1)
                #Es:  torch.Size([1, 11, 3, 480, 854])
                #Es[:,:,t]:  torch.Size([1, 11, 480, 854])
                #Ms:  torch.Size([1, 11, 3, 480, 854])
                
                # update
                keys, values = this_keys, this_values
                #print('########### t: ', t)
                #input('Press Enter button to continue...')
                
            ##########################
            # parei aqui, nao sei onde deve ficar a loss... =(
            loss = loss + criterion(Es[:,:,1:2], Ms[:,:,1:2].float())              
                
            if loss > 0:  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # logging and display
            #if (seq+1) % args.disp_interval == 0:
            if (seq+1) % 10 == 0:
                writer.add_scalar('Train/BCE', loss, seq + epoch * iters_per_epoch)
                #writer.add_scalar('Train/IOU', iou(torch.cat((1-all_E, all_E), dim=1), all_M), i + epoch * iters_per_epoch)
                print('loss: {}'.format(loss))
                
            
            ("Fim do loop: {}/{} ".format(seq,iters_per_epoch))
            #input('Press Enter button to continue...')
            
        if True: #epoch % 10 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch,
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                       },
                       save_name)   
    
    
        
if __name__ == "__main__":
    
    main()
    
# for m in range(3):
#     plt.matshow(Ms[0,0,m,:,:])
#     plt.show()
#     input('Press Enter button to continue...')
#     plt.matshow(Es[0,0,m,:,:])
#     plt.show()
#     input('Press Enter button to continue...')
    
    
           