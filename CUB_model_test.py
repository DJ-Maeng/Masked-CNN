# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 03:42:31 2021

@author: maeng
"""

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torchvision import datasets


##################################################################################################################
#                                                                                                                #
#                                               Parameter                                                        #
#                                                                                                                #
################################################################################################################## 

alias = "VGG16"

DATA_PATH = "D:/data/CUB_200_2011/"
SAVE_PATH = os.path.join(DATA_PATH, "record", alias)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS     = 200
BATCH_SIZE = 25
NUM_CLASS  = 200

##################################################################################################################
#                                                                                                                #
#                                              Data Load                                                         #
#                                                                                                                #
##################################################################################################################

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

#transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
#dataset    = datasets.ImageFolder(root      = os.path.join(DATA_PATH, "train"), transform = transform)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size = 10, shuffle = False)

# mean = 0.
# std = 0.
# nb_samples = 0.
# for i, _ in dataloader:
#     batch_samples = i.size(0)
#     i = i.view(batch_samples, i.size(1), -1)
#     mean += i.mean(2).sum(0)
#     std += i.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples #[0.4831, 0.4918, 0.4248]
# std /= nb_samples  #[0.1839, 0.1833, 0.1943]

transform_train = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.4831, 0.4918, 0.4248], [0.1839, 0.1833, 0.1943])
])

transform_test = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.4831, 0.4918, 0.4248], [0.1839, 0.1833, 0.1943]),
])

trainset = datasets.ImageFolder(root      = os.path.join(DATA_PATH, "train"),
                                transform = transform_train)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)

testset = datasets.ImageFolder(root      = os.path.join(DATA_PATH, "test"),
                               transform = transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=False)

##################################################################################################################
#                                                                                                                #
#                                                NetWork                                                         #
#                                                                                                                #
##################################################################################################################

model = torchvision.models.vgg16_bn(pretrained = True)
model.classifier[-1] = nn.Linear(in_features = model.classifier[-1].in_features, out_features = NUM_CLASS, bias = True)

#model = torchvision.models.resnet50(pretrained = True)
#model.fc = nn.Linear(in_features = model.fc.in_features, out_features = NUM_CLASS, bias = True)

# model = torchvision.models.googlenet(pretrained = True)
# model.fc = nn.Linear(in_features = 1024, out_features = NUM_CLASS, bias = True)

model = model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 40, gamma=0.5)

##################################################################################################################
#                                                                                                                #
#                                                  Train                                                         #
#                                                                                                                #
##################################################################################################################
train_loss = dict()
test_loss  = dict()
test_acc   = dict()

train_num = len(trainloader.dataset)
test_num  = len(testloader.dataset)

check = int(len(trainset) / BATCH_SIZE) + 1
check = list(np.arange(0, check, check / 10).astype(int)) + [check - 1]

#epoch = 1
for epoch in range(1, EPOCHS + 1):
    
    start = datetime.datetime.now()
    
    model.train()
    
    lr = scheduler.get_last_lr()[0]
    
    count = 0

    for batch_idx, (data, target) in enumerate(trainloader):
        
        data, target = data.to(DEVICE), target.to(DEVICE)
       
        optimizer.zero_grad()
       
        output = model(data)

        loss = F.cross_entropy(output, target)
       
        loss.backward()
        optimizer.step()

        
        count += len(data)
        
        if batch_idx in check:            
            print("Train Epoch : {}  [{:5}/{:5} ({:6.2f}%)] Loss : {:.6f}".format(epoch, count, train_num, count / train_num * 100, loss.item()))
            
    scheduler.step()
            
    train_loss[epoch] = loss.item()
    
    end = datetime.datetime.now()
    time =  (end - start).total_seconds()
    
    print("Train Epoch : {}  Train Time : {:.2f}s  lr : {}".format(epoch, time, lr))
    
##################################################################################################################
#                                                                                                                #
#                                                   Test                                                         #
#                                                                                                                #
##################################################################################################################    
    #if epoch % 5 == 0:
    
    start = datetime.datetime.now()
    
    model.eval()
    
    loss = 0
    correct = 0
    
    with torch.no_grad():    
        
        for data, target in testloader:
            
            data, target = data.to(DEVICE), target.to(DEVICE)        
            
            output = model(data)
            
            loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            
            correct += pred.eq(target.view_as(pred)).sum().item()
        
    loss /= len(testloader.dataset)
    
    acc = 100. * correct / len(testloader.dataset)
        
    test_loss[epoch] = loss
    test_acc[epoch]  = acc
    
    end = datetime.datetime.now()
    time =  (end - start).total_seconds()
    
    print('Train Epoch : {}  Test Time  : {:.2f}s  Acc: {:.2f}%  Loss : {:.4f}'.format(epoch, time, acc, loss))
    print("   ")
    
    if epoch % 5 == 0:
                
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, alias + "_" + str(epoch) + ".pth"))
        
        with open(os.path.join(SAVE_PATH, alias + "_" + str(epoch) + "_train_loss.txt"), 'w') as f:
            f.write(str(train_loss))
            f.close()
        with open(os.path.join(SAVE_PATH, alias + "_" + str(epoch) + "_test_loss.txt"), 'w') as f:
            f.write(str(test_loss))
            f.close()
        with open(os.path.join(SAVE_PATH, alias + "_" + str(epoch) + "_test_acc.txt"), 'w') as f:
            f.write(str(test_acc))
            f.close()

##################################################################################################################
#                                                                                                                #
#                                               Plotting                                                         #
#                                                                                                                #
##################################################################################################################

last = str(max(set([int(i.split(".")[0].split("_")[1]) for i in os.listdir(os.path.join(SAVE_PATH))])))
train_loss = eval(open(os.path.join(SAVE_PATH, alias + "_" + last + "_train_loss.txt"), 'r').read())
test_loss  = eval(open(os.path.join(SAVE_PATH, alias + "_" + last + "_test_loss.txt"), 'r').read())
test_acc  = eval(open(os.path.join(SAVE_PATH, alias + "_" + last + "_test_acc.txt"), 'r').read())

plt.plot(train_loss.values())
plt.plot(test_loss.values())
plt.show()

plt.plot(test_acc.values())
plt.show()
##################################################################################################################
#                                                                                                                #
#                                             Evaluation                                                         #
#                                                                                                                #
##################################################################################################################
epoch = max(test_acc.items(), key = lambda x : x[1])[0]
print("Best All   Epoch : " + str(epoch) + "  Acc : " + str(test_acc[epoch]))

epoch = max([i for i in test_acc.items() if i[0] % 5 == 0], key = lambda x : x[1])[0]
print("Best Save  Epoch : " + str(epoch) + "  Acc : " + str(test_acc[epoch]))

model = torchvision.models.vgg16_bn(pretrained = False)
model.classifier[-1] = nn.Linear(in_features = model.classifier[-1].in_features, out_features = NUM_CLASS, bias = True)
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, alias + "_" + str(epoch) + ".pth")))
model = model.to(DEVICE)

# model = torchvision.models.resnet50(pretrained = False)
# model.fc = nn.Linear(in_features = model.fc.in_features, out_features = NUM_CLASS, bias = True)
# model.load_state_dict(torch.load(os.path.join(SAVE_PATH, alias + "_" + str(epoch) + ".pth")))
# model = model.to(DEVICE)

#eval(open(os.path.join(DATA_PATH, "record", "GOOGLENET_" + str(100) + "_test_acc.txt"), 'r').read())
# model = torchvision.models.googlenet(pretrained = True)
# model.fc = nn.Linear(in_features = 1024, out_features = NUM_CLASS, bias = True)
# model.load_state_dict(torch.load(os.path.join(DATA_PATH, "record", "GOOGLENET_" + str(95) + ".pth")))
# model = model.to(DEVICE)


predict = pd.DataFrame()

model.eval()

with torch.no_grad():
    
    for i, (data, target) in enumerate(testloader):
    
        data = data.cuda(DEVICE, non_blocking=True)
        target = target.cuda(DEVICE, non_blocking=True)

        # compute output
        output = model(data)
        
        pred              = pd.DataFrame(output.tolist())
        pred["target"]    = target.tolist() 
        pred["top1_pred"] = output.topk(1)[1].t().squeeze().tolist()
        pred["top5_pred"] = output.topk(5)[1].tolist()
        pred["topk_pred"] = output.topk(NUM_CLASS)[1].tolist()
        print(i)
        
        predict = pd.concat([predict, pred], sort = False)

predict["img"] = [i[0].split("\\")[-1] for i in testloader.dataset.samples]
predict["path"] = [i[0] for i in testloader.dataset.samples]

predict = predict[["img", "path"] + list(predict.columns)[:-2]]
predict = predict.reset_index(drop = True)

sum(predict.top1_pred == predict.target) / len(predict)

predict.to_csv("D:/masked-CNN/CUB_prop/CUB_img_" + alias + "_prop.csv", encoding = "UTF-8-sig", index = False)




        

