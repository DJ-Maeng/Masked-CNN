# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:22:08 2021

@author: maeng
"""
import os
import copy
import datetime
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T

def softmax(x, dim, numpy = True):
    if numpy == 1:
        return F.softmax(torch.tensor(x), dim = dim).numpy()
    else:
        return F.softmax(torch.tensor(x), dim = dim)

def interpolate(x, size = (224, 224)):
    x = torch.tensor(x).unsqueeze(0)
    x = F.interpolate(x, size = size, mode='bilinear', align_corners=False)
    return x.squeeze()

def load_image(image_path):   
    return Image.open(image_path).convert('RGB')

##################################################################################################################
#                                                                                                                #
#                                               Parameter                                                        #
#                                                                                                                #
################################################################################################################## 

transform_test = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.4831, 0.4918, 0.4248], [0.1839, 0.1833, 0.1943]),
])

transform_show = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

alias = "VGG16"

DATA_PATH  = "D:/data/CUB_200_2011"
LOAD_PATH  = "D:/masked-CNN/CUB_prop/"
SAVE_PATH  = os.path.join("D:/masked-CNN/CUB_mask", alias)
MODEL_PATH = os.path.join(DATA_PATH, "record", alias)

TARGET_LAYER = 42
NUM_CLASS = 200

##################################################################################################################
#                                                                                                                #
#                                                    Data Load                                                   #
#                                                                                                                #
################################################################################################################## 

#Prop Data
data = pd.read_csv(os.path.join(LOAD_PATH, "CUB_img_" + alias + "_prop.csv"), encoding = "UTF-8-sig")

data["top1_correct"] = data.top1_pred == data.target
data["top5_correct"] = data.apply(lambda x : x.target in eval(x.top5_pred), axis = 1)

sum(data.top1_correct) / len(data)
sum(data.top5_correct) / len(data)

idx = [str(i) for i in range(0, NUM_CLASS)]

data["top1_prop"] = softmax(data[idx].values, dim = 1, numpy = False).topk(1)[0].sum(dim = 1).tolist()

#Best epoch
last      = str(max(set([int(i.split(".")[0].split("_")[1]) for i in os.listdir(os.path.join(MODEL_PATH))])))
test_acc  = eval(open(os.path.join(MODEL_PATH, alias + "_" + last + "_test_acc.txt"), 'r').read())

epoch = max([i for i in test_acc.items() if i[0] % 5 == 0], key = lambda x : x[1])[0]
print("Best Save  Epoch : " + str(epoch) + "  Acc : " + str(test_acc[epoch]))

#VGG
model = torchvision.models.vgg16_bn(pretrained = False)
model.classifier[-1] = nn.Linear(in_features = 4096, out_features = 200, bias = True)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, alias + "_" + str(epoch) + ".pth")))
model = model.to(DEVICE)

#RESNET
# model = torchvision.models.resnet50(pretrained = False)
# model.fc = nn.Linear(in_features = model.fc.in_features, out_features = NUM_CLASS, bias = True)
# model.load_state_dict(torch.load(os.path.join(MODEL_PATH, alias + "_" + str(epoch) + ".pth")))
# model = model.to(DEVICE)

model.eval()
##################################################################################################################
#                                                                                                                #
#                                                   Define Weight                                                #
#                                                                                                                #
################################################################################################################## 
def weight(x, deg):
    K = len(x)
    p = (len(x) + 1) / 2
    #p = len(x) + 1
    w = list()
    for i in range(1, len(x) + 1):

        if i <= p:
            w.append(np.power((1/(1 - p)) * (i - p), deg))
        else:
            w.append(np.power(-1, deg + 1) * np.power((1 / (p - K)) * (i - p), deg))

    return torch.tensor(w)

def weight(x, num_class):
    return F.softmax(torch.tensor(x.values.astype(float)), dim = 0).topk(num_class)[0]


#plt.plot(weight(idx, 2))


##################################################################################################################
#                                                                                                                #
#                                               Probability Correction                                           #
#                                                                                                                #
################################################################################################################## 
th_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999]
r_list  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

w = weight(idx, 2).unsqueeze(-1).unsqueeze(-1)

evaluation = list()
i = 18
for i in range(0, len(data)):

    start = datetime.datetime.now()
    
    with open(os.path.join(SAVE_PATH, "CUB_" + alias + "_img_" + str(i) + ".pickle"), "rb") as file:
        file = pickle.load(file)    
    
    picture = load_image(data.path[i])
    img     = transform_test(picture).unsqueeze(0).to(DEVICE)
    target  = data.loc[i, "target"]
    original = F.softmax(torch.tensor(data.iloc[i][idx].values.astype(float)), dim = 0)
    
    w = weight(data.loc[i, idx], NUM_CLASS).unsqueeze(-1).unsqueeze(-1)
    
    saliency_map = interpolate(file["saliency_map"])    
    saliency_map = (saliency_map * w).sum(dim = 0)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    
    #plt.imshow(saliency_map)
    
    val = torch.sort(saliency_map.flatten(), descending = True)[0]
    
    pred = list()
    for j in th_list:
        #j = 0.3
        th = val[int(saliency_map.shape[0] * saliency_map.shape[1] * j)]
        
        mask = saliency_map.clone()
        mask[mask < th] = 0
        #mask[mask >= th] = 1
        mask = mask.unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                
        #Visualize        
        #plt.imshow(transform_show(picture).permute(1,2,0))
        #plt.imshow(mask.squeeze().cpu(), vmin = mask.min(), vmax = mask.max())
        #plt.imshow(np.array(transform_show(picture).permute(1,2,0)) * mask.squeeze().unsqueeze(-1).repeat(1,1,3).cpu().numpy())

        with torch.no_grad():
            output = model(img * mask).cpu()
        
        correction = F.softmax(output[0], dim = 0)
    
        #correction.topk(5)
        #original.topk(5)
    
        for k in r_list:
    
            final = (original + (correction * k)) / (1 + k)
    
            final = final.topk(5)[1].tolist()
                
            pred.append(int(final[0]))        
            pred.append(int(final[0]) == target)
    
    evaluation.append(pred)
    
    end = datetime.datetime.now()
    print(i, (end - start).total_seconds())
    
evaluation = pd.DataFrame(evaluation)
evaluation.columns = sum([["th_" + str(i) + "_r_" + str(j) + "_pred", "th_" + str(i) + "_r_" + str(j) + "_correct"] for i in th_list for j in r_list], [])

evaluation["top1_correct"] = data.top1_correct
evaluation["top1_prop"]    = data.top1_prop

result = pd.DataFrame(index = th_list, columns = r_list)
for i in th_list:
    #i = 0.3; j = 0.6
    for j in r_list:

        tab = pd.crosstab(evaluation.top1_correct, evaluation["th_" + str(i) + "_r_" + str(j) + "_correct"])
        #print(pd.crosstab(data1["top5_correct"].iloc[start : end], data1["cam_top5_correct" + "_" + str(l)].iloc[start : end]))
        result.loc[i, j] = tab.iloc[0, 1] - tab.iloc[1, 0]        

result


ex = evaluation[(evaluation["top1_correct"] == 1) & (evaluation["th_0.3_r_0.6_correct"] == 0)][["top1_prop", "top1_correct", "th_0.3_r_0.6_correct"]]
plt.hist(ex.top1_prop)
