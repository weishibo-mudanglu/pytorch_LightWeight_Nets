import os
import cv2
import numpy as np
import mmcv
import torch
import time
from pathlib import Path
#/home/industai/Downloads/拟真测试集-正异-已清洗-eval_full_sensitivity_get_detection_result_with_cluster_TP-FP_no_deeplearning_fr        epoch_40_0.99

# std = np.array([57.0, 57.0, 57.0])
class ClassifyModel():
    def __init__(self,model,device='cpu') -> None:
        self.model = torch.jit.load(model)
        self.model.eval()
        self.device=device
        self.model.to(device)
        self.mean = np.array([103.53, 116.28, 123.675])
        self.std = np.array([57.375, 57.12, 58.395])

    def do_infer(self,img,model_type="single") -> list:
        img=self.img_preprocess(img)
        with torch.no_grad():
            out = self.model(img.to(self.device))
            return self.postprocess(out,model_type)

    def img_preprocess(self,img,size=(224,224)):

        img = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv,img)  # inplace
        img = img.transpose(2,0,1)
        img = img[None,:,:,:]
        img=torch.from_numpy(img)
        img = img.to(torch.float32)
        return img
    def postprocess(self,output,result_type):
        if result_type=="single":#普通分类模型，返回一个一维列表[类别一置信度，类别二置信度]
            
            return output
        if result_type=="multi":#多属性分类，返回一个二维列表[[类别一置信度，类别二置信度，～～]，[属性一置信度，属性二置信度，～～]]
            conf1=0.0
            conf2=0.0
            attribute1=0.0
            attribute2=0.0
            return [[conf1,conf2],[attribute1,attribute2]]
if __name__=='__main__':
    model="/home/industai/code_folder/Python_code/gitlab_project/pytorch_LightWeight_Nets/cnn.torchscript"
    device="cuda:0"
    img = cv2.imread("/home/industai/code_folder/Python_code/gitlab_project/pytorch_LightWeight_Nets/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
    classify1=ClassifyModel(model,device)
    out = classify1.do_infer(img,model_type="single")
    print(out)


