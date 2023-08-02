import torch
import dataset
import os
import pandas as pd
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import resnet50,resnet18
# import vgg16 
#colorID = {"1":yellow,"2":"brown","3":"green","4":"silver","5":"red","6":"blue","7":"white","8":"gray","9":"caffe","10":"black"}
#typeID = {"1":"taxi","2":"offroadvehicle","3":"minibus","4":"salooncar","5":,"6":"pickup","7":"bus","8":"truck","9":"SUV"}
#direction_attrs = {"0":"front","1":"rear"}
color_attrs = ["yellow", "orange","green","gray","red","blue","white","golden","brown","black"]
direction_attrs = ["taxi","offroadvehicle","minibus","salooncar","pickup","bus","truck","SUV"]
type_attrs = ["front","rear"]
# print(torch.cuda.get_device_name())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if __name__ == '__main__':
if 0:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.5
    EPOCH = 2
    N_CLASSES = 5
    INPUTSIZE = 128
    WORKERS=4
    #对图片的预处理操作
    trans_form = transforms.Compose([
        transforms.RandomResizedCrop(INPUTSIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
        ])
    train_path = [
                  "/media/industai/DATA/data/animals/classify-animals/cane.txt",
                  "/media/industai/DATA/data/animals/classify-animals/cavallo.txt",
                  "/media/industai/DATA/data/animals/classify-animals/elefante.txt",
                  "/media/industai/DATA/data/animals/classify-animals/farfalla.txt",
                  "/media/industai/DATA/data/animals/classify-animals/gallina.txt"
                  ]
    # test_path = ["/media/industai/DATA/data/变电站异常分类/normal/内江/35kV东区变电站/val.txt","/media/industai/DATA/data/变电站异常分类/abnormal/内江/35kV东区变电站/val.txt"]
    train_data=dataset.MyDataset(txt=train_path, transform=trans_form)
    test_data=dataset.MyDataset(txt=train_path, transform=trans_form)
    trainLoader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,num_workers=WORKERS)
    testLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False,num_workers=WORKERS)

    net = resnet50.Classifier(num_cls = N_CLASSES,input_size=INPUTSIZE)
    net.to(device)
    # vgg16.cuda()

    # Loss, Optimizer & Scheduler
    #binary_cross_entropyloss()二值交叉熵损失函数，可以用于多标签分类
    loss_func = torch.nn.CrossEntropyLoss()#两个参数，第一个是输出结果，第二个是标签数字（自动转换为独热码）
    # cost = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.03, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_acc_data = []
    # Train the model
    for epoch in range(EPOCH):

        avg_loss = 0
        cnt = 0
        net.train()
        for images, labels in trainLoader:
            images = images.to(device)
            labels = labels.to(device)
            # labels = labels.long()
            # labels = torch.tensor(labels,dtype=torch.float32)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(images)# .squeeze(-1)

            loss_name = loss_func(outputs, labels[:,0])
            # loss_state = loss_func(outputs[:, 1], labels[:, 1])
            # loss = loss_name + loss_state
            loss = loss_name
            avg_loss += loss.data
            cnt += 1

            # print("[E: %d] loss_color: %f,loss_direction:%f, avg_loss: %f" % (epoch, loss_name.data,loss_state.data, avg_loss/cnt))
            if (cnt%100==0):
                print("[E: %d] loss_color: %f,avg_loss: %f" % (epoch, loss_name.data, avg_loss/cnt))
            loss.backward()
            optimizer.step()
            # scheduler.step(avg_loss)
        
        net.eval()
        tp,fp,tn,fn=0,0,0,0
        total=0
        for images, labels in testLoader:
            images = images.to(device)
            # labels = labels.to(device)
            outputs = net(images)
            predicted = resnet50.get_predict(outputs)    
              
            total += labels.size(0)
            for index in range(predicted.size(0)):
                marix_out = predicted[index].numpy()
                marix_label = labels[index].numpy()
                if marix_label==marix_out:
                    tn=tn+1
                else:
                    fn=fn+1

        print("acc: %f" % (100* (tn)/total))
            
            # correct += (predicted == labels).sum()
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

if __name__ == '__main__':
    net=resnet50.Classifier(num_cls = 5,input_size=128)
    net.load_state_dict(torch.load("cnn.pkl"))
    example_input = torch.randn(1,3,128,128)
    traced_model = torch.jit.trace(net,example_inputs=example_input)
    traced_model.save("cnn.torchscript")
    torch.save(net, 'cnn.pt')

        

