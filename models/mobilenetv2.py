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
def PDP_block(channel_in,channel_out,k_size,stride_num,padding_num):
    layer = nn.Sequential(
        nn.Conv2d(channel_in,channel_in*6,1,stride_num,0),
        nn.BatchNorm2d(channel_in*6),
        nn.ReLU6(),
        nn.Conv2d(channel_in*6,channel_in*6,k_size,stride_num,padding_num,groups=channel_in),
        nn.BatchNorm2d(channel_in*6),
        nn.ReLU6(),
        nn.Conv2d(channel_in*6,channel_out,1,stride_num,0),
        nn.BatchNorm2d(channel_out),

    )
    return layer
class mobilenetv2(nn.Module):
    def __init__(self,n_classes=1000):
        super(mobilenetv2,self).__init__()
        #v2 block
        self.layer1 = PDP_block(3,64,3,1,1)
        self.layer2 = PDP_block(64,64,3,1,1)

        self.layer3 = PDP_block(64,64,3,2,0)

        self.layer4 = PDP_block(64,128,3,1,1)
        self.layer5 = PDP_block(128,128,3,1,1)

        self.layer6 = PDP_block(128,128,3,2,0)

        self.layer7 = PDP_block(128,256,3,1,1)
        self.layer8 = PDP_block(256,256,3,1,1)
        self.layer9 = PDP_block(256,256,3,1,1)

        self.layer10 = PDP_block(256,256,3,2,0)

        self.layer11 = PDP_block(256,512,3,1,1)
        self.layer12 = PDP_block(512,512,3,1,1)
        self.layer13 = PDP_block(512,512,3,1,1)

        self.layer14 = PDP_block(512,512,3,2,0)

        self.layer15 = PDP_block(512,512,3,1,1)
        self.layer16 = PDP_block(512,512,3,1,1)
        self.layer17 = PDP_block(512,512,3,1,1)

        self.layer18 = PDP_block(512,512,3,2,0)
        #fc
        self.layer19 = vgg_fc_layer(7*7*512,4096)
        self.layer20 = vgg_fc_layer(4096,4096)
        self.layer21 = nn.Linear(4096,n_classes)
    def forward(self,x):
        out = self.layer1(x)
        out = out+ self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out + self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out + self.layer8(out)
        out = out + self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = out + self.layer12(out)
        out = out + self.layer13(out)
        out = self.layer14(out)
        out = out + self.layer15(out)
        out = out + self.layer16(out)
        out = out + self.layer17(out)
        feature = self.layer18(out)#卷积层结束，reshape变成一维张量进入全连接层，输出特征
        out = feature.view(out.size(0),-1)
        out = self.layer19(out)
        out = self.layer20(out)
        out = self.layer21(out)
        return feature,out