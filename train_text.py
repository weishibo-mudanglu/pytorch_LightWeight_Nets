import torch
import dataset
import os
import pandas as pd
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import Squeezcnet
import resnet18
# import vgg16 
#colorID = {"1":yellow,"2":"brown","3":"green","4":"silver","5":"red","6":"blue","7":"white","8":"gray","9":"caffe","10":"black"}
#typeID = {"1":"taxi","2":"offroadvehicle","3":"minibus","4":"salooncar","5":,"6":"pickup","7":"bus","8":"truck","9":"SUV"}
#direction_attrs = {"0":"front","1":"rear"}
color_attrs = ["yellow", "orange","green","gray","red","blue","white","golden","brown","black"]
direction_attrs = ["taxi","offroadvehicle","minibus","salooncar","pickup","bus","truck","SUV"]
type_attrs = ["front","rear"]
# print(torch.cuda.get_device_name())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    BATCH_SIZE = 20
    LEARNING_RATE = 0.5
    EPOCH = 5
    N_CLASSES = 20
    INPUTSIZE = 224
    #对图片的预处理操作
    trans_form = transforms.Compose([
        transforms.RandomResizedCrop(INPUTSIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
        ])
    train_path = r"D:\work\datas\vehicle_attri\dataset\VeRi\image_train"
    test_path = r"D:\work\datas\vehicle_attri\dataset\VeRi\image_test"
    train_data=dataset.MyDataset(txt='train_label.txt',path = train_path, transform=trans_form)
    test_data=dataset.MyDataset(txt='test_label.txt',path = test_path, transform=trans_form)
    # trainData = dsets.ImageFolder('D:/work/py/pytorch_work/flowers/train', transform)
    # testData = dsets.ImageFolder('D:/work/py/pytorch_work/flowers/text', transform)
    trainLoader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
    # trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    # testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)
    # vgg16 = vgg16.VGG16(n_classes=N_CLASSES)
    net = resnet18.Classifier(num_cls = N_CLASSES,input_size=INPUTSIZE)
    net.to(device)
    # vgg16.cuda()

    # Loss, Optimizer & Scheduler
    #binary_cross_entropyloss()二值交叉熵损失函数，可以用于多标签分类
    loss_func = torch.nn.CrossEntropyLoss()#两个参数，第一个是输出结果，第二个是标签数字（自动转换为独热码）
    # cost = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_acc_data = []
    # Train the model
    for epoch in range(EPOCH):

        avg_loss = 0
        cnt = 0
        net.train()
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)# .squeeze(-1)
            # outputs = outputs.squeeze(-1)


            labels = labels.long()
            # label = torch.tensor(labels, dtype=torch.long)
            loss_direction = loss_func(outputs[:,:2], labels[:, 0])
            loss_type = loss_func(outputs[:, 2:10], labels[:, 1])
            loss_color = loss_func(outputs[:, 10:], labels[:, 2])
            loss = loss_color + loss_direction + 2.0 * loss_type  # greater weight to type
            avg_loss += loss.data
            cnt += 1

            print("[E: %d] loss_color: %f,loss_direction:%f,loss_type:%f avg_loss: %f" % (epoch, loss_color.data,loss_direction.data,loss_type.data, avg_loss/cnt))
            # print("[E: %d] loss_direction:%f, avg_loss: %f" % (epoch,loss_direction.data, avg_loss/cnt))
            # loss.backward()
            loss_direction.backward()
            optimizer.step()
            scheduler.step(avg_loss)
        
        net.eval()
        correct = 0
        color_correct = 0
        direction_correct = 0
        type_correct = 0
        total = 0
        for images, labels in testLoader:
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            predicted = resnet18.get_predict(outputs)    
              
            total += labels.size(0)
            for index in range(predicted.size(0)):
                marix_out = predicted[index].numpy()
                marix_label = labels[index].numpy()
                if(all(marix_out==marix_label)):
                    correct = correct + 1
                if(marix_out[0] == marix_label[0]):
                    direction_correct = direction_correct+1
                if(marix_out[1] == marix_label[1]):
                    type_correct = type_correct+1
                if(marix_out[2] == marix_label[2]):
                    color_correct = color_correct+1
            
            # correct += (predicted == labels).sum()
            print("avg acc: %f,color acc: %f,direction acc: %f,type acc: %f" % (100* correct/total,100* color_correct/total,100* direction_correct/total,100* type_correct/total))
            # print("color acc: %f" % (100* color_correct/total))
            # print("direction acc: %f" % (100* direction_correct/total))
            # print("type acc: %f" % (100* type_correct/total))
        # loss_acc_data.append([avg_loss/cnt,correct/total,direction_correct/total,type_correct/total,color_correct/total])
    
    # Test the model
    

    # for images, labels in testLoader:
    #     # images = images.cuda()
    #     outputs = net(images)
    #     predicted = Squeezcnet.get_predicted(outputs)
        
    #     total += labels.size(0)
    #     correct += (predicted.cpu() == labels).sum()
    #     print(predicted, labels, correct, total)
    #     print("avg acc: %f" % (100* correct/total))

    # Save the Trained Model
    # col_name = ['平均损失','全属性','方向','类型','颜色']
    # df = pd.DataFrame(columns=col_name, data=loss_acc_data)
    # df.to_csv("stu_info.csv", encoding='utf-8', index=False)

    torch.save(net.state_dict(), 'cnn.pkl')

        

