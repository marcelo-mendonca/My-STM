# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:18:20 2020

@author: Marcelo
"""
from torch.utils import data
from dataset import DAVIS_MO_Test

DATA_ROOT = 'E:/Py_all/rvos-master/databases/DAVIS2017'
YEAR = 16
SET = 'val'

print('Parte 1')
Testset = DAVIS_MO_Test(DATA_ROOT, resolution='480p', imset='20{}\\{}.txt'.format(YEAR,SET), single_object=(YEAR==16))

try:
    print(Testset.image_dir)
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    print( 'Num videos: ', len(Testset.videos))
    print('batch size: ', Testloader.batch_size)
except:
    print('deu ruim!')

#Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
print('Parte 2')
for seq, V in enumerate(Testloader):
    print('Parte For')
    if seq < 1:
        print('Parte if')
        Fs, Ms, num_objects, info = V
        seq_name = info['name'][0]
        num_frames = info['num_frames'][0].item()
    else:
        print('Parte else')
        break
print('Parte fim')
