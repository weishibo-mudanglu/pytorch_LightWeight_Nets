import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random
import cv2

def find_suffix(name):
    for suffix in ['.png','.jpg','.jpeg','.bmp']:
        index = name.find(suffix)
        if index!=-1:
            return index+len(suffix)
    raise KeyError("图片名包含未知后缀")
#以torch.utils.data.Dataset为基类创建MyDataset
class MyDataset(Dataset):
    #stpe1:初始化
    def __init__(self, txt, transform=None, target_transform=None):
        self.imgs=[]
        if isinstance(txt,list):
            for item in txt:
                fh = open(item, 'r')#打开标签文件
                for line in fh:#遍历标签文件每行
                    line = line.rstrip()#删除字符串末尾的换行符
                    index_pic=find_suffix(line)
                    pic_path = line[:index_pic]
                    templist = [int(item) for item in line[index_pic+1:].split(" ")]

                    self.imgs.append((pic_path,templist))#把图片名words[0]，标签int(words[1])放到imgs里
        
                self.transform = transform
                self.target_transform = target_transform    
        else:
            fh = open(txt, 'r')#打开标签文件
            for line in fh:#遍历标签文件每行
                line = line.rstrip()#删除字符串末尾的换行符
                index_pic=find_suffix(line)
                pic_path = line[:index_pic]
                templist = [int(item) for item in line[index_pic+1:].split(" ")]

                self.imgs.append((pic_path,templist))#把图片名words[0]，标签int(words[1])放到imgs里
            self.transform = transform
            self.target_transform = target_transform
 
    def __getitem__(self, index):#检索函数
        fn, label = self.imgs[index]#读取文件名、标签
        label = np.array(label)

        img = Image.open(fn).convert('RGB')#通过PIL.Image读取图片
        if self.transform is not None:
            img = self.transform(img)
        return img,label
 
    def __len__(self):
        return len(self.imgs)

class SimameseDataset(Dataset):
    #stpe1:初始化
    def __init__(self, txt, transform=None, target_transform=None):
        self.imgs=[]
        if isinstance(txt,list):
            for item in txt:
                fh = open(item, 'r',encoding='UTF-8')#打开标签文件
                for line in fh:#遍历标签文件每行
                    line = line.rstrip()#删除字符串末尾的换行符
                    index_pic=find_suffix(line)
                    pic_path = line[:index_pic]
                    templist = [int(item) for item in line[index_pic+1:].split(" ")]

                    self.imgs.append((pic_path,templist))#把图片名words[0]，标签int(words[1])放到imgs里
        
                self.transform = transform
                self.target_transform = target_transform    
        else:
            fh = open(txt, 'r')#打开标签文件
            for line in fh:#遍历标签文件每行
                line = line.rstrip()#删除字符串末尾的换行符
                index_pic=find_suffix(line)
                pic_path = line[:index_pic]
                templist = [int(item) for item in line[index_pic+1:].split(" ")]

                self.imgs.append((pic_path,templist))#把图片名words[0]，标签int(words[1])放到imgs里
            self.transform = transform
            self.target_transform = target_transform
 
    def __getitem__(self, index):#检索函数
        fn, label = self.imgs[index]#读取文件名、标签
        label = np.array(label)
        # if label ==0:
        #     label=1.0
        # else:
        #     label=-1.0
        label = np.array(label).astype(np.float32)
        img = cv2.imread(fn)
        W = int(img.shape[1]/2)
        img1 = img[:,0:W,:]
        img2 = img[:,W:W*2,:]
        img1 = Image.fromarray(img1) 
        img2 = Image.fromarray(img2) 
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2,label
 
    def __len__(self):
        return len(self.imgs)
