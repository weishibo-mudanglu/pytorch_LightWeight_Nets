import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random

 
#以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    #stpe1:初始化
    def __init__(self, txt,path, transform=None, target_transform=None,):
        fh = open(txt, 'r')#打开标签文件
        imgs = []#创建列表，装东西
        for line in fh:#遍历标签文件每行
            line = line.rstrip()#删除字符串末尾的空格
            words = line.split()#通过空格分割字符串，变成列表
            templist = []
            templist.append(int(words[1])-1)
            templist.append(int(words[2])-1)
            templist.append(int(words[3])-1)
            if(int(words[2]) == 1 and random.random()>0.5):continue
            imgs.append((words[0],templist))#把图片名words[0]，标签int(words[1])放到imgs里
        
        self.path = path
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):#检索函数
        fn, label = self.imgs[index]#读取文件名、标签
        label = np.array(label)
        complete_path = os.path.join(self.path,fn)

        img = Image.open(complete_path).convert('RGB')#通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)
