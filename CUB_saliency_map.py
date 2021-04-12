# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:46:22 2021

@author: maeng
"""
import os
import numpy as np
import datetime
import pickle
import gc
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

os.chdir("D:/masked-CNN")
from cam import scorecam_vggnet_batch
from cam import scorecam_resnet_batch
from cam import scorecam_vggnet_original

def load_image(image_path):   
    return Image.open(image_path).convert('RGB')

def softmax(x):
    return F.softmax(torch.tensor(x), dim = 0).numpy()

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

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

alias = "VGG16"

DATA_PATH  = "D:/data/CUB_200_2011"
LOAD_PATH  = "D:/masked-CNN/CUB_prop/"
SAVE_PATH  = os.path.join("D:/masked-CNN/CUB_mask", alias)
MODEL_PATH = os.path.join(DATA_PATH, "record", alias)

TARGET_LAYER = 42
NUM_CLASS = 200
BATCH_SIZE = 50

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

#Best epoch
last = str(max(set([int(i.split(".")[0].split("_")[1]) for i in os.listdir(os.path.join(MODEL_PATH))])))
test_acc  = eval(open(os.path.join(MODEL_PATH, alias + "_" + last + "_test_acc.txt"), 'r').read())
epoch = max(test_acc.items(), key = lambda x : x[1])[0]
print("Best All   Epoch : " + str(epoch) + "  Acc : " + str(test_acc[epoch]))

epoch = max([i for i in test_acc.items() if i[0] % 5 == 0], key = lambda x : x[1])[0]
print("Best Save  Epoch : " + str(epoch) + "  Acc : " + str(test_acc[epoch]))

#VGG
model = torchvision.models.vgg16_bn(pretrained = False)
model.classifier[-1] = nn.Linear(in_features = model.classifier[-1].in_features, out_features = 200, bias = True)
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
#                                                    Mask image                                                  #
#                                                                                                                #
################################################################################################################## 
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

i = 0
for i in range(0, len(data)):

    start = datetime.datetime.now()

    picture = load_image(data.path[i])
    img     = transform_test(picture).unsqueeze(0).to(DEVICE)
    label   = torch.tensor(eval(data.topk_pred[i]))
    
    #score_cam1    = scorecam_vggnet_original.ScoreCam(model, TARGET_LAYER)
    #saliency_map1 = score_cam1.generate_cam(img, label[0])

    score_cam    = scorecam_vggnet_batch.ScoreCam(model, TARGET_LAYER)
    saliency_map = score_cam.generate_cam(img, label, BATCH_SIZE)
    
    meta = dict()
    meta["img"]         = data.img[i]
    meta["path"]        = data.path[i]
    meta["target"]      = data.target[i]
    meta["origin_pred"] = data.top1_pred[i]
    meta["mask_label"]  = label.cpu().numpy()
    meta["saliency_map"] = saliency_map
    
    with open(os.path.join(SAVE_PATH, "CUB_" + alias + "_img_" + str(i) + ".pickle"), 'wb') as file:
        pickle.dump(meta, file)
        file.close()
    
    del meta, file
    gc.collect()
    
    end = datetime.datetime.now()
    
    print(i, (end - start).total_seconds())


