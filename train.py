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
import tqdm
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter

### My libs
from dataset import DAVIS_MO_Test, DAVIS_MO_Train, DAVIS_MO_Val
from model import STM
from helpers import *

# Constants
MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)
NUM_EPOCHS = 1000
inputs_to_gpu = True


def main():
    print(">>>> Hello world! \nMy STM training script starting...")
    
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
                      default=3, type=int)
    
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
    Trainloader = data.DataLoader(Trainset, batch_size=4, shuffle=False, num_workers=2)
    
    Valset = DAVIS_MO_Val(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    Valloader = data.DataLoader(Valset, batch_size=12, shuffle=False, num_workers=4)
    
    Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=1)
    
    # Isntantiate the model
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
    
    # loop for training and validation
    for epoch in range(start_epoch, args.num_epochs):
        
        # validating ##################################
        if (epoch+1) % args.eval_epoch == 0:

            with torch.no_grad():
                print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
                model.eval()
                new_validate(model, criterion, Valloader, inputs_to_gpu, device, Mem_every=5, Mem_number=None)
                print("terminou o validate")
                input("Press Enter button to continue...")
                #pbar = tqdm.tqdm(total=len(Valloader))
                for seq, V in enumerate(Valloader):
                    #pbar.update(1)
                    
                    ############# interrupção só para testar
                    if seq > 4:
                        break
                    
                    Fss, Mss, nums_objects, _ = V
                    #Fss:  torch.Size([4, 3, 1, 100, 100])
                    #Mss:  torch.Size([4, 11, 1, 100, 100])
                    #nums_objects:  tensor([[1],[1],[1],[1]])

                    if torch.cuda.is_available() and inputs_to_gpu:
                        Fss = Fss.to(device)
                        Mss = Mss.to(device)
                        nums_objects = nums_objects.to(device)
                        
                    batch_size = int(Fss.size(0))                    
                    for batch_idx in range(batch_size):                
                        
                        #Fs, Ms = Fss[batch_idx], Mss[batch_idx]
                        Fs, Ms = Fss[batch_idx,:,:,0:99,0:99], Mss[batch_idx,:,:,0:99,0:99]
                        Fs = torch.unsqueeze(Fs, dim=0)
                        Ms = torch.unsqueeze(Ms, dim=0)
                        num_objects = nums_objects[batch_idx]
                        #num_frames = Fs.size(2)
                        num_frames = 10
                        
                        run_validate(model, Fs, Ms, num_frames, num_objects, criterion, Mem_every=5, Mem_number=None)
                        #model = STM()
                        #Fs:  torch.Size([1, 3, 69, 480, 910])
                        #Ms:  torch.Size([1, 11, 69, 480, 910])
                        #num_frames: 69
                        #num_objects:  2

                #pbar.close()
                      
        # end of testing ###########################                
                
        
        # training
        model.eval() #set eval mode to disable batchnorm and dropout
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        
        #maximum_skip is increased by 5 at every 20 epoch during main-training
        #Trainset.frame_skip initial value is 5
        if (epoch > 0) and (epoch % 20 == 0):
            Trainset.frame_skip = min([Trainset.frame_skip+5, 25])
          
        for seq, V in enumerate(Trainloader):
            
            ############# interrupção só para testar
            if seq > 4:
                break
            
            Fss, Mss, nums_objects, info = V

            #info:  {'name': ['cat-girl', 'classic-car', 'lady-running', 'swing'], 
            #'num_frames': tensor([89, 63, 65, 60]), 
            #'size_480p': [tensor([100, 100, 100, 100]), tensor([100, 100, 100, 100])]}
            
            # send input tensors to gpu
            if torch.cuda.is_available() and inputs_to_gpu:
                Fss = Fss.to(device)
                Mss = Mss.to(device)
                nums_objects = nums_objects.to(device)
            
            optimizer.zero_grad()
            
            tt = time.time()
            
            #Es recebe de Ms a máscara com menor indice temporal; as outras posições de Es ficam vazias
            #Fs:  torch.Size([1, 3, 3, 480, 854])
            #Ms:  torch.Size([1, 11, 3, 480, 854])
            #Es:  torch.Size([1, 11, 3, 480, 854])
            #Fs:  torch.Size([1, 3, 3, 384, 384])
            #Ms:  torch.Size([1, 11, 3, 384, 384])
            #num_objects:  torch.Size([1, 1])
            
            loss = 0
            
            batch_size = int(Fss.size(0))
            for batch_idx in range(batch_size):                
                
                Fs, Ms = Fss[batch_idx], Mss[batch_idx]
                Fs = torch.unsqueeze(Fs, dim=0)
                Ms = torch.unsqueeze(Ms, dim=0)
                num_objects = nums_objects[batch_idx]
                Es = torch.zeros_like(Ms)
                Es[:,:,0] = Ms[:,:,0]
            
                #loop over the 3-1 frame+annotation samples (1st frame is reference frame)
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
                    
                    # compute loss
                    loss += criterion(Es[:,:,t].clone(), Ms[:,:,t].float()) 
                    
                print('batch: {}/{} '.format(batch_idx+1, batch_size))
            
            # divive loss by Nº frames = 2 (t=1 and t=2; t=0 is reference frame) * batch size
            loss /= 2*batch_size
            
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
                
            
            print("Fim do loop: {}/{} ".format(seq,iters_per_epoch))
            
            
        if epoch % 10 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch,'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),}, save_name)
            print('Model saved in: {}'.format(save_name))
        
        print("The End")
    

def run_validate(model, Fs, Ms, num_frames, num_objects, criterion, Mem_every=None, Mem_number=None):
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

    loss = 0
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
        
        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
            #(t=39) logit: torch.Size([1, 11, 480, 910])
        
        Es[:,:,t] = F.softmax(logit, dim=1)
        #(t=39) Es: torch.Size([1, 11, 69, 480, 910])
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
        # compute loss
        loss += criterion(Es[:,:,t].clone(), Ms[:,:,t].float())
        
    loss /= num_frames-1
    print('val loss: {}'.format(loss))
    

def new_validate(model, criterion, Valloader, inputs_to_gpu, device, Mem_every=None, Mem_number=None):
    #model = STM()
    #Fs:  torch.Size([1, 3, 69, 480, 910])
    #Ms:  torch.Size([1, 11, 69, 480, 910])
    #num_frames: 69
    #num_objects:  2
    
    idx = 0
    next_change = 0
    to_second_frame = False
    
    for seq, V in enumerate(Valloader):
                    
        ############# interrupção só para testar
        #if seq > 4:
        #    break

        
        Fss, Mss, nums_objects, infos = V
        nums_frames = infos['num_frames']
        #Fss:  torch.Size([4, 3, 1, 100, 100])
        #Mss:  torch.Size([4, 11, 1, 100, 100])
        #nums_objects:  tensor([[1],[1],[1],[1]])
        #infos:  {'name': ['bear', 'bear', 'bear', 'bear'], 
        #       'num_frames': tensor([82, 82, 82, 82]), 
        #       'size_480p': [tensor([100, 100, 100, 100]), tensor([100, 100, 100, 100])]}    
        
        if torch.cuda.is_available() and inputs_to_gpu:
            Fss = Fss.to(device)
            Mss = Mss.to(device)
            nums_objects = nums_objects.to(device)
            
        batch_size = int(Fss.size(0))                    
        for batch_idx in range(batch_size):                
            
            Fs, Ms = Fss[batch_idx,:,0], Mss[batch_idx,:,0]
            Fs = torch.unsqueeze(Fs, dim=0)
            Ms = torch.unsqueeze(Ms, dim=0)
            num_objects = nums_objects[batch_idx]
            num_frames = nums_frames[batch_idx].item()
            
            # New sequence begins
            if idx == next_change:
                next_change += num_frames
                if Mem_every:
                    to_memorize = [int(i) for i in np.arange(idx, next_change, step=Mem_every)]
                elif Mem_number:
                    to_memorize = [int(round(i)) for i in np.linspace(idx, next_change, num=Mem_number+2)[:-1]]
                else:
                    raise NotImplementedError
                #Se mem_every=5, então to_memorize = [0, 5, 10, 15...]
                loss = 0
                
                first_frame = True
                second_frame = False
                to_second_frame = True
            elif to_second_frame:
                first_frame = False
                second_frame = True
                to_second_frame = False
            else:
                first_frame = False
                second_frame = False
                to_second_frame = False        
            
            if not first_frame:
        
                # memorize
                with torch.no_grad():
                    prev_key, prev_value = model(prev_Fs, prev_Ms, torch.tensor([num_objects]))
                    #prev_key(k4):  torch.Size([1, 11, 128, 1, 30, 57])
                    #prev_value(v4):  torch.Size([1, 11, 512, 1, 30, 57])
         
                if second_frame: # 
                    this_keys, this_values = prev_key, prev_value # only prev memory
                else:
                    this_keys = torch.cat([keys, prev_key], dim=3)
                    this_values = torch.cat([values, prev_value], dim=3)                

                # segment
                with torch.no_grad():
                    logit = model(Fs, this_keys, this_values, torch.tensor([num_objects]))
                    #(t=39) logit: torch.Size([1, 11, 480, 910])
                
                    pred = F.softmax(logit, dim=1)
                    #(t=39) Es: torch.Size([1, 11, 69, 480, 910])
                    
                    # compute loss
                    loss += criterion(pred, Ms.float())
                
                # update
                if second_frame or idx in to_memorize:
                    keys, values = this_keys, this_values
                
            prev_Fs = Fs
            prev_Ms = Ms
            idx += 1       
            print("idx: {}, seq: {}/{}, batch_idx: {}, num_frames: {} ".format(idx, seq,len(Valloader), batch_idx, num_frames))
            
        if idx == next_change:
            loss /= num_frames-1
            print('val loss: {}'.format(loss))
        
    
        
if __name__ == "__main__":
    
    main()
    
# for m in range(3):
#     plt.matshow(Ms[0,0,m,:,:])
#     plt.show()
#     input('Press Enter button to continue...')
#     plt.matshow(Es[0,0,m,:,:])
#     plt.show()
#     input('Press Enter button to continue...')
    
    
           