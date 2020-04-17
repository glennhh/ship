#!/usr/bin/env python

import os

#folder = '/home/cloudlab/Data/ml/dataset/train_v2'    
#saveFile =  '/home/cloudlab/Data/ml/ship/scr/input/imgList.csv'
folder = '/home/hao/Data/course/ML/project/train_v2'
saveFile =  '/home/hao/Data/course/ML/project/ship/scr/input/imgList.csv'
dirList = os.listdir(folder) 

try:
    os.remove(saveFile)
except FileNotFoundError:
    pass

with open(saveFile, 'a') as a:
    for img in dirList:
        a.writelines(img+ '\n') 
    a.close()  











