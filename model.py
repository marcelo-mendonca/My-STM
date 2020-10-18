from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
# general libs
import math
from helpers import *
 
class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride==1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
 
        if self.downsample is not None:
            x = self.downsample(x)
         
        return x + r 

class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim
        #f:  torch.Size([2, 3, 480, 912])
        #m:  torch.Size([2, 1, 480, 912])
        #o:  torch.Size([2, 1, 480, 912])        

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
 
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, mdim): #mdim=256
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1) 
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        #r4(m4): torch.Size([2, 1024, 30, 57])
        #r3(r3e): torch.Size([2, 512, 60, 114])
        #r2(r2e): torch.Size([2, 256, 120, 228])
        m4 = self.ResMM(self.convFM(r4))
        #m4:  torch.Size([2, 256, 30, 57])
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        #m3:  torch.Size([2, 256, 60, 114])
        m2 = self.RF2(r2, m3) # out: 1/4, 256
        #m2:  torch.Size([2, 256, 120, 228])

        p2 = self.pred2(F.relu(m2))
        #p2:  torch.Size([2, 2, 120, 228])
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        #p:  torch.Size([2, 2, 480, 912])
        return p #, p2, p3, p4


#Memory Read
class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()
        
    # entradas: Mem_key, Mem_value, Query_key, Query_value
    def forward(self, m_in, m_out, q_in, q_out):  # m_in: o,c,t,h,w
        #[bike-packing]: num_frames: 40, num_objects: 2
        #m_in:  torch.Size([2, 128, 1, 30, 57])
        #m_out:  torch.Size([2, 512, 1, 30, 57])
        #q_in:  torch.Size([2, 128, 30, 57])
        #q_out:  torch.Size([2, 512, 30, 57])
        B, D_e, T, H, W = m_in.size()  
        #m_in size:  torch.Size([2, 128, 1, 30, 57]) -> #B=2, D_e=128, T=1, H=30, W=57
        _, D_o, _, _, _ = m_out.size() 
        #m_out size:  torch.Size([2, 512, 1, 30, 57]) -> #D_o=512

        mi = m_in.view(B, D_e, T*H*W) 
        #mi view:  torch.Size([2, 128, 1710])
        mi = torch.transpose(mi, 1, 2)  # b, THW, emb        
        #mi transpose:  torch.Size([2, 1710, 128])
 
        qi = q_in.view(B, D_e, H*W)  # b, emb, HW
        #qi view:  torch.Size([2, 128, 1710])
        
        #bmm -> batch matrix-matrix multiplication
        p = torch.bmm(mi, qi) # b, THW, HW
        #p bmm:  torch.Size([2, 1710, 1710])
        p = p / math.sqrt(D_e)
        #p sqrt:  torch.Size([2, 1710, 1710])
        p = F.softmax(p, dim=1) # b, THW, HW
        #p softmax:  torch.Size([2, 1710, 1710])

        mo = m_out.view(B, D_o, T*H*W) 
        #mo:  torch.Size([2, 512, 1710])
        mem = torch.bmm(mo, p) # Weighted-sum B, D_o, HW
        #mem bmm:  torch.Size([2, 512, 1710])
        mem = mem.view(B, D_o, H, W)
        #mem view:  torch.Size([2, 512, 30, 57])

        mem_out = torch.cat([mem, q_out], dim=1)
        #mem_out:  torch.Size([2, 1024, 30, 57])
        #p:  torch.Size([2, 1710, 1710])

        return mem_out, p

class KeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)
 
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class STM(nn.Module):
    def __init__(self):
        super(STM, self).__init__()
        self.Encoder_M = Encoder_M() 
        self.Encoder_Q = Encoder_Q() 

        self.KV_M_r4 = KeyValue(1024, keydim=128, valdim=512)
        self.KV_Q_r4 = KeyValue(1024, keydim=128, valdim=512)

        self.Memory = Memory()
        self.Decoder = Decoder(256)
 
    def Pad_memory(self, mems, num_objects, K):
        pad_mems = []
        for mem in mems:
            pad_mem = ToCuda(torch.zeros(1, K, mem.size()[1], 1, mem.size()[2], mem.size()[3]))
            pad_mem[0,1:num_objects+1,:,0] = mem
            pad_mems.append(pad_mem)
        return pad_mems

    def memorize(self, frame, masks, num_objects):
        #frame:  torch.Size([1, 3, 480, 910])
        #masks:  torch.Size([1, 11, 480, 910])
        #num_objects shape:  torch.Size([1])
        
        # memorize a frame 
        num_objects = num_objects[0].item()

        _, K, H, W = masks.shape # B = 1
        #num_objects item:  2
        # K=11, H=480, W=912

        (frame, masks), pad = pad_divide_by([frame, masks], 16, (frame.size()[2], frame.size()[3]))
        #faz as medidas do frames serem divisíveis por 16
        # 480 é divisível então mantém; 910 passa pra 912
        #frame:  torch.Size([1, 3, 480, 912])
        #masks:  torch.Size([1, 11, 480, 912])
        #pad:  (1, 1, 0, 0)

        # make batch arg list
        B_list = {'f':[], 'm':[], 'o':[]}
        for o in range(1, num_objects+1): # 1 - no
            B_list['f'].append(frame)
            B_list['m'].append(masks[:,o])            
            B_list['o'].append((torch.sum(masks[:,1:num_objects+1], dim=1).clamp(0,1)))
            #B_list['o'].append((torch.sum(masks[:,0:o], dim=1) + \
            #    torch.sum(masks[:,o:num_objects+1], dim=1)).clamp(0,1))
            
            #B_list['f'][0]:  torch.Size([1, 3, 384, 384])
            #B_list['m'][0]:  torch.Size([1, 384, 384])
            #B_list['o'][0]:  torch.Size([1, 384, 384])            
            #len(B_list):  3
            #len(B_list['f']):  2
            #len(B_list['m']):  2
            #len(B_list['o']):  2                     

        # make Batch
        B_ = {}
        for arg in B_list.keys(): #arg recebe 'f', 'm' e 'o'
            B_[arg] = torch.cat(B_list[arg], dim=0)
            #B_[f]:  torch.Size([2, 3, 480, 912]) -> 2 é o num_objects
            #B_[m]:  torch.Size([2, 480, 912])
            #B_[o]:  torch.Size([2, 480, 912])
        
        r4, _, _, _, _ = self.Encoder_M(B_['f'], B_['m'], B_['o'])
        #r4:  torch.Size([2, 1024, 30, 57])
        k4, v4 = self.KV_M_r4(r4) # num_objects, 128 and 512, H/16, W/16
        #k4 -> key M:  torch.Size([2, 128, 30, 57]) -> 480/16=30, 912/16=57
        #v4 -> Value M:  torch.Size([2, 512, 30, 57])
        k4, v4 = self.Pad_memory([k4, v4], num_objects=num_objects, K=K)
        #k4:  torch.Size([1, 11, 128, 1, 30, 57])
        #v4:  torch.Size([1, 11, 512, 1, 30, 57])
        return k4, v4

    #Soft_aggregation recebe um mapa de probabilidades pra cada objeto (K vai
    #até 11, mas só o nº de canais correspondente ao nº de objetos tem valores).
    #A função então faz 1-ps pra pegar a probabilidade do background. Então junta
    #tudo em logits, de modo que cada pixel tem formato [1, K, H, W]. No canal
    #K=0, pixel de BG=1 e FG=0. Em K=1, BG=0 e pixels do 1º objeto = 1. Em K=2,
    #BG=0, 2º objeto = 1, e assim sucessivamente.
    def Soft_aggregation(self, ps, K):
        #ps:  torch.Size([2, 480, 912])
        #K = 11
        num_objects, H, W = ps.shape #num_objects = 2, H = 480, W = 912
        em = ToCuda(torch.zeros(1, K, H, W)) 
        #em zeros:  torch.Size([1, 11, 480, 912])
        em[0,0] =  torch.prod(1-ps, dim=0) # bg prob
        #em prod:  torch.Size([1, 11, 480, 912])
        em[0,1:num_objects+1] = ps # obj prob
        #em num_obj:  torch.Size([1, 11, 480, 912])
        em = torch.clamp(em, 1e-7, 1-1e-7)
        #em clamp:  torch.Size([1, 11, 480, 912])
        logit = torch.log((em /(1-em)))
        #logit:  torch.Size([1, 11, 480, 912])
        return logit

    def segment(self, frame, keys, values, num_objects):
        #[bike-packing]: num_frames: 40, num_objects: 2
        #frame: torch.Size([1, 3, 480, 910])
        #keys: torch.Size([1, 11, 128, 1, 30, 57])
        #values: torch.Size([1, 11, 512, 1, 30, 57])
        #num_objects: torch.Size([1]) = 2
        
        num_objects = num_objects[0].item()
        _, K, keydim, T, H, W = keys.shape # B=1, K=11, Keydim=128, T=1, H=480, W=910  
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
        #faz as medidas do frames serem divisíveis por 16
        # 480 é divisível então mantém; 910 passa pra 912
        #frame:  torch.Size([1, 3, 480, 912])
        #pad:  (1, 1, 0, 0)

        r4, r3, r2, _, _ = self.Encoder_Q(frame)
        #r4:  torch.Size([1, 1024, 30, 57])
        #r3:  torch.Size([1, 512, 60, 114])
        #r2:  torch.Size([1, 256, 120, 228])
        
        k4, v4 = self.KV_Q_r4(r4)   # 1, dim, H/16, W/16
        #k4:  torch.Size([1, 128, 30, 57])
        #v4:  torch.Size([1, 512, 30, 57])
        
        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(num_objects,-1,-1,-1), v4.expand(num_objects,-1,-1,-1)
        #k4e:  torch.Size([2, 128, 30, 57])
        #v4e:  torch.Size([2, 512, 30, 57])
        r3e, r2e = r3.expand(num_objects,-1,-1,-1), r2.expand(num_objects,-1,-1,-1)
        #r3e:  torch.Size([2, 512, 60, 114])
        #r2e:  torch.Size([2, 256, 120, 228])
        
        # memory select kv:(1, K, C, T, H, W)
        m4, viz = self.Memory(keys[0,1:num_objects+1], values[0,1:num_objects+1], k4e, v4e)
        #m4:  torch.Size([2, 1024, 30, 57])
        #viz:  torch.Size([2, 1710, 1710])
        logits = self.Decoder(m4, r3e, r2e)
        #logits:  torch.Size([2, 2, 480, 912])
        ps = F.softmax(logits, dim=1)[:,1] # no, h, w  
        #ps = indipendant possibility to belong to each object
        #ps:  torch.Size([2, 480, 912])
        
        logit = self.Soft_aggregation(ps, K) # 1, K, H, W
        #logit:  torch.Size([1, 11, 480, 912])

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]
            #logit if2:  torch.Size([1, 11, 480, 910])
            
        return logit    

    def forward(self, *args, **kwargs):
        if args[1].dim() > 4: # keys
            return self.segment(*args, **kwargs)  #retorna logit
            #logit:  torch.Size([1, 11, 480, 910])
        else:
            return self.memorize(*args, **kwargs) #retorna k4, v4
            #k4:  torch.Size([1, 11, 128, 1, 30, 57])
            #v4:  torch.Size([1, 11, 512, 1, 30, 57])
        
        #[bike-packing]: num_frames: 40, num_objects: 2
        #segment:
        #arg shape:  torch.Size([1, 3, 480, 910])          arg dim:  4
        #arg shape:  torch.Size([1, 11, 128, 1, 30, 57])   arg dim:  6 (keys)
        #arg shape:  torch.Size([1, 11, 512, 1, 30, 57])   arg dim:  6
        #arg shape:  torch.Size([1]) = num_objects         arg dim:  1
        #memorize
        #arg shape:  torch.Size([1, 3, 480, 910])          arg dim:  4
        #arg shape:  torch.Size([1, 11, 480, 910])         arg dim:  4
        #arg shape:  torch.Size([1]) = num_objects         arg dim:  1


#Garbage
#plt.matshow(B_['o'].permute(1,2,0)[:,:,0])
#plt.show()
#input("Press Enter to continue...")
