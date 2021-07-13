import torch
import os
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer
    
def DPW_block(channel_in,channel_out,k_size,stride_num,padding_num):
    layer = nn.Sequential(
        nn.Conv2d(channel_in,channel_in,k_size,stride_num,padding_num,groups=channel_in),
        nn.BatchNorm2d(channel_in), #归一化
        nn.ReLU6(),
        nn.Conv2d(channel_in,channel_out,1,stride_num,0),
        nn.BatchNorm2d(channel_out), #归一化
        nn.ReLU6()

    )
    return layer
class mobilenetv1(nn.Module):
    def __init__(self,n_classes=1000):
        super(mobilenetv1,self).__init__()
        #conv block
        self.layer1 = DPW_block(3,64,3,1,1)
        self.layer2 = DPW_block(64,64,3,1,1)
        self.layer3 = nn.MaxPool2d(2,2)
        self.layer4 = DPW_block(64,128,3,1,1)
        self.layer5 = DPW_block(128,128,3,1,1)
        self.layer6 = nn.MaxPool2d(2,2)
        self.layer7 = DPW_block(128,256,3,1,1)
        self.layer8 = DPW_block(256,256,3,1,1)
        self.layer9 = DPW_block(256,256,3,1,1)
        self.layer10 = nn.MaxPool2d(2,2)
        self.layer11 = DPW_block(256,512,3,1,1)
        self.layer12 = DPW_block(512,512,3,1,1)
        self.layer13 = DPW_block(512,512,3,1,1)
        self.layer14 = nn.MaxPool2d(2,2)
        self.layer15 = DPW_block(512,512,3,1,1)
        self.layer16 = DPW_block(512,512,3,1,1)
        self.layer17 = DPW_block(512,512,3,1,1)
        self.layer18 = nn.MaxPool2d(2,2)
        #FC layers
        self.layer19 = vgg_fc_layer(7*7*512,4096)
        self.layer20 = vgg_fc_layer(4096,4096)
        self.layer21 = nn.Linear(4096,n_classes)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        feature = self.layer18(out)#卷积层结束，reshape变成一维张量进入全连接层，输出特征
        out = feature.view(out.size(0),-1)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.layer21(out)
        return feature,out