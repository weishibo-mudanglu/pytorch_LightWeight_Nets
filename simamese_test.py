import torch
import dataset
import os
import pandas as pd
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import simamese_resnet50
import numpy as np
import matplotlib.pyplot as plt
# import vgg16 
#colorID = {"1":yellow,"2":"brown","3":"green","4":"silver","5":"red","6":"blue","7":"white","8":"gray","9":"caffe","10":"black"}
#typeID = {"1":"taxi","2":"offroadvehicle","3":"minibus","4":"salooncar","5":,"6":"pickup","7":"bus","8":"truck","9":"SUV"}
#direction_attrs = {"0":"front","1":"rear"}
type_attrs = ["normal","abnormal"]
# print(torch.cuda.get_device_name())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
# if 0:
    BATCH_SIZE = 1
    EPOCH = 1
    INPUTSIZE = 224
    WORKERS=4
    PreModel="/home/industai/code_folder/Python_code/gitlab_project/CLIP/RN50.torchscript"
    #对图片的预处理操作
    trans_form = transforms.Compose([
        transforms.Resize((INPUTSIZE,INPUTSIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
        ]) 
    
    train_path = [
                # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---35kV田家变电站/train.txt',
            # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---35kV东区变电站/train.txt',
            # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---35kV越溪变电站/train.txt',
            # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---35kV富溪变电站/train.txt',
            # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---110kV沙坝变电站/train.txt',
            # '/media/industai/DATA/data/变电站异常分类/normal_no_error/逐张标记---110kV大洲变电站/train.txt',
            # "/media/industai/DATA/data/变电站异常分类/摆拍站点测试数据/GT/范家坡.txt",
            # "/media/industai/DATA/data/变电站异常分类/摆拍站点测试数据/TP/范家坡.txt",
            # "/media/industai/DATA/data/变电站异常分类/逐框标注数据/abnormal/35kV凌家变电站例行巡视_0727_异常/val.txt",
            # "/media/industai/DATA/data/变电站异常分类/逐框标注数据/abnormal/35kV凌家变电站例行巡视/train.txt",
            # "/home/industai/datasets/异常分类模型数据/测试保存数据/v59/研发测试集-evalfullsensitivitygetdetectionresultwithclusterTP-FPnodeeplearningfrom两河76onlyaligntobasebigsunallimgcamerainfomanyreport24expand80defaultth0.4bigsun0.8_epoch_108/GT外1/变电站摆排-异常摆排-all-正异-异常常规.txt",
            # "/home/industai/datasets/异常分类模型数据/测试保存数据/v59/研发测试集-evalfullsensitivitygetdetectionresultwithclusterTP-FPnodeeplearningfrom两河76onlyaligntobasebigsunallimgcamerainfomanyreport24expand80defaultth0.4bigsun0.8_epoch_108/GT外0/变电站摆排-异常摆排-all-正异-异常常规.txt",
            # "/home/industai/datasets/异常分类模型数据/测试保存数据/v59/研发测试集-evalfullsensitivitygetdetectionresultwithclusterTP-FPnodeeplearningfrom两河76onlyaligntobasebigsunallimgcamerainfomanyreport24expand80defaultth0.4bigsun0.8_epoch_108/GT内0/变电站摆排-异常摆排-all-正异-异常常规.txt",
            # "/home/industai/datasets/异常分类模型数据/测试保存数据/v59/研发测试集-evalfullsensitivitygetdetectionresultwithclusterTP-FPnodeeplearningfrom两河76onlyaligntobasebigsunallimgcamerainfomanyreport24expand80defaultth0.4bigsun0.8_epoch_108/GT内1/变电站摆排-异常摆排-all-正异-异常常规.txt",
            "/media/industai/DATA/data/变电站异常分类/temp/正常.txt"
                  ]
    # test_path = ["/media/industai/DATA/data/变电站异常分类/normal/内江/35kV东区变电站/val.txt","/media/industai/DATA/data/变电站异常分类/abnormal/内江/35kV东区变电站/val.txt"]
    plt.figure()
    for index,path in enumerate(train_path):
        path=[path]
        train_data=dataset.SimameseDataset(txt=path, transform=trans_form)
        test_data=dataset.SimameseDataset(txt=path, transform=trans_form)
        trainLoader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=WORKERS)
        testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=WORKERS)

        net = simamese_resnet50.Simamese(input_size=INPUTSIZE,PreModel=PreModel)
        net.to(device)
        for epoch in range(EPOCH):

            avg_loss = 0
            cnt = 0

            net.eval()
            tp,fp,tn,fn=0,0,0,0
            total=0
            result=[]
            for images1,images2, labels in testLoader:
                images1 = images1.to(device).half()
                images2 = images2.to(device).half()
                # labels = labels.to(device)
                outputs = net(images1,images2).to('cpu')
                result.extend(outputs.detach().numpy())
                print(labels)
                print(outputs)
            unique, counts = np.unique((np.array(result)*100).astype('int64'), return_counts=True)

            plt.plot(unique,counts/np.sum(counts),label=str(index))
    plt.legend()
    plt.show()
    print("1")
# if __name__ == '__main__':
#     net=resnet50.Classifier(num_cls = 5,input_size=128)
#     net.load_state_dict(torch.load("cnn.pkl"))
#     example_input = torch.randn(1,3,128,128)
#     traced_model = torch.jit.trace(net,example_inputs=example_input)
#     traced_model.save("cnn.torchscript")
#     torch.save(net, 'cnn.pt')
