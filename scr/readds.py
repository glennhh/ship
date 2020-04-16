#!/usr/bin/env python

import os

folder = '/home/cloudlab/Data/ml/dataset/train_v2'    
#folder = '/home/cloudlab/Data/ml/ship/scr'
saveFile =  '/home/cloudlab/Data/ml/ship/scr/input/imgList.csv'
dirList = os.listdir(folder) 

try:
    os.remove(saveFile)
except FileNotFoundError:
    pass

with open(saveFile, 'a') as a:
    for img in dirList:
        a.writelines(img+ '\n') 
    a.close()  











