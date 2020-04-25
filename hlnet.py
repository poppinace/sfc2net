# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 09:49:51 2018

@author: poppinace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hlblock import *
import numpy as np
from mixnet.model_all import tf_mixnet_l_myself,tf_mixnet_l_myself_fusion,tf_mixnet_s_myself, tf_mixnet_l
import torchvision.models as models

class Mixnet_l_designed_model       (nn.Module):
    def __init__(self,max_class_number=80,**kwargs):
        super(Mixnet_l_designed_model, self).__init__()
        self.upsample=Up_sample_mixnet_l()
        
        # prediction layers
        self.classification = nn.Sequential(
            nn.AvgPool2d((4, 4), stride=(1, 1)),
            nn.Conv2d(104, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, max_class_number, 1)
        )
        
    def forward(self, x8,x16,x32): 
        x8,x16,x32=self.upsample(x8,x16,x32)
                
        x8=self.classification(x8)
        return x8
        
class Mixnet_l_classification_fusion(nn.Module):
    def __init__(self,freeze_bn=False,class2regression=None,**kwargs):
        super(Mixnet_l_classification_fusion, self).__init__()
        self.class2regression=class2regression
        max_class_number=len(    class2regression)
        self.input_size=32
        self.output_stride=8
            
        self.class2regression=torch.FloatTensor(self.class2regression).cuda()
        self.backbone=tf_mixnet_l_myself_fusion()
        if freeze_bn:
            print("Fix BN!")
            self.freeze_bn()
        self.designed_model=  Mixnet_l_designed_model(max_class_number)
     
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m,nn.BatchNorm2d):
                m.eval()
    def forward(self, x): 
        
        imH, imW = x.size()[2:]
        x2,x4,x8,x16,x32=self.backbone(x)
        x8=self.designed_model(x8,x16,x32)
        
        if not self.training:  
            x=x8.data.max(1)[1]
            n,h,w=x.size()
            x=torch.index_select(self.class2regression,0,x.long().view(n*h*w))
            x=x.view(n,h,w)
            x=x.unsqueeze_(0)
            xcount=x.clone()
            
            _, _, h, w = x.size()            
            accm = torch.cuda.FloatTensor(1, self.input_size*self.input_size, h*w).fill_(1)           
            accm = F.fold(accm, (imH, imW), kernel_size=self.input_size, stride=self.output_stride)
            accm = 1 / accm
            accm /= self.input_size**2
            accm = F.unfold(accm, kernel_size=self.input_size, stride=self.output_stride).sum(1).view(1, 1, h, w)
            x *= accm
            
            return x,xcount
            
        return x8
        
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                        m.weight, 
                        mode='fan_in', 
                        nonlinearity='relu'
                        )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)          
                
class Up_sample_mixnet_l(nn.Module):
    def __init__(self):
        super(Up_sample_mixnet_l, self).__init__()
        
        self.layer32_16_1 = nn.Conv2d(in_channels=1536, out_channels=208, kernel_size=1, padding=0) 
        self.layer32_16_2 = nn.BatchNorm2d(208)
        self.layer32_16_3 = nn.ReLU(inplace=True)
        self.layer32_16_4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer32_16_5 = nn.Conv2d(in_channels=208, out_channels=208, kernel_size=1, padding=0) 
        self.layer32_16_6 = nn.BatchNorm2d(208)
        self.layer32_16_7 = nn.ReLU(inplace=True)
        self.layer32_16_8 = nn.Conv2d(in_channels=104, out_channels=104, kernel_size=1, padding=0) 
        self.layer32_16_9 = nn.BatchNorm2d(104)
        self.layer32_16_10 = nn.ReLU(inplace=True)
        self.layer32_16_11 = nn.Conv2d(in_channels=104+208, out_channels=208, kernel_size=1, padding=0) 
        self.layer32_16_12 = nn.BatchNorm2d(208)
        self.layer32_16_13 = nn.ReLU(inplace=True)
                    
        self.layer16_8_1 = nn.Conv2d(in_channels=208, out_channels=104, kernel_size=1, padding=0) 
        self.layer16_8_2 = nn.BatchNorm2d(104)
        self.layer16_8_3 = nn.ReLU(inplace=True)
        self.layer16_8_4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.layer16_8_5 = nn.Conv2d(in_channels=104, out_channels=104, kernel_size=1, padding=0) 
        self.layer16_8_6 = nn.BatchNorm2d(104)
        self.layer16_8_7 = nn.ReLU(inplace=True)
        self.layer16_8_8 = nn.Conv2d(in_channels=56, out_channels=56, kernel_size=1, padding=0) 
        self.layer16_8_9 = nn.BatchNorm2d(56)
        self.layer16_8_10 = nn.ReLU(inplace=True)
        self.layer16_8_11 = nn.Conv2d(in_channels=56+104, out_channels=104, kernel_size=1, padding=0) 
        self.layer16_8_12 = nn.BatchNorm2d(104)
        self.layer16_8_13 = nn.ReLU(inplace=True)
        
    def forward(self, x8,x16,x32):
        x=self.layer32_16_1 (x32)
        x=self.layer32_16_2 (x)
        x=self.layer32_16_3 (x)
        x=self.layer32_16_4 (x)
        x=self.layer32_16_5 (x)
        x=self.layer32_16_6 (x)
        x=self.layer32_16_7 (x)
        x16=self.layer32_16_8 (x16)
        x16=self.layer32_16_9 (x16)
        x16=self.layer32_16_10 (x16)
        
        x16=torch.cat((x, x16), 1)
        x16=self.layer32_16_11 (x16)
        x16=self.layer32_16_12 (x16)
        x16=self.layer32_16_13 (x16)
        
        x = self.layer16_8_1(x16)
        x = self.layer16_8_2(x)
        x = self.layer16_8_3(x)
        x = self.layer16_8_4(x)
        x = self.layer16_8_5(x)
        x = self.layer16_8_6(x)
        x = self.layer16_8_7(x)        
        x8 = self.layer16_8_8(x8)      
        x8 = self.layer16_8_9(x8)      
        x8 = self.layer16_8_10(x8)
        
        x8=torch.cat((x, x8), 1)
        x8=self.layer16_8_11 (x8)
        x8=self.layer16_8_12 (x8)
        x8=self.layer16_8_13 (x8)
        del x
        
        return x8,x16,x32
        
def Mixnet_l_backbone(pretrained=False,model_name='mixnet_fusion_classification',\
                      max_class_number=80,step_log=0.1, start_log=-2,input_size=32,output_stride=8,freeze_bn=False,**kwargs):
    class2regression=np.zeros(max_class_number)
    for i in range(1,max_class_number):
        if i==1:
            lower=0
        else:
            lower=np.exp((i-2)*step_log+start_log)
        upper=np.exp((i-1)*step_log+start_log)
        class2regression[i]=(lower+upper)/2
                    
    net=Mixnet_l_classification_fusion(freeze_bn=freeze_bn,class2regression=class2regression,**kwargs)

    return net
