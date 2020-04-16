#!/usr/bin/env python

import os
import sys
import glob
import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from skimage.exposure import histogram
from skimage import io, measure
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel, rank  
from skimage.segmentation import watershed
from skimage.morphology import watershed, disk   
from skimage.util import img_as_ubyte  
import pandas as pd
from seg import segmentation as seg 


curPath = os.path.dirname(os.path.realpath(__file__))  
outputPath = curPath + '/output/' 
recordFile = outputPath + 'record.csv'
finalModelFile = outputPath + 'model.csv'     
predictFile    = outputPath + 'predict.csv'  
trainPath = curPath + '/input/train/'
fileList = glob.glob(trainPath + "*.jpg")
imgsOri = io.ImageCollection(fileList)
trueMasks = pd.read_csv(curPath + '/input/train_ship_segmentations_v2.csv')  
trueMasks.head()

#############################
###    Constants          ###
#############################
patchHoleSize = 20
shape=(768, 768)  
iouThrh = 0.5       # IoU treshhold 

def show1(img):
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def show2(img1, name1, img2, name2):
    plt.figure(figsize=(6,3))
    plt.subplot(121).set_title(name1)
    plt.imshow(img1)
    plt.subplot(122).set_title(name2)
    plt.imshow(img2)
    #plt.imshow(img2, cmap=plt.cm.Spectral, interpolation='nearest') 
    plt.show(block=False)
    plt.pause(9)
    plt.close()
    
def show4(img1, name1, img2, name2, img3, name3, img4, name4):
    plt.close('all')
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout()
    ax = plt.subplot(221)
    ax.set_title(name1)
    plt.imshow(img1)
    ax = plt.subplot(222)
    ax.set_title(name2)
    plt.imshow(img2)
    ax = plt.subplot(223)
    ax.set_title(name3)
    plt.imshow(img3)
    ax = plt.subplot(224)
    ax.set_title(name4)
    plt.imshow(img4)
    plt.show(block=False)
    plt.pause(9)
    plt.close()

def prep():
    os.remove(recordFile) 
    with open(recordFile, 'w'):
        pass
    os.remove(finalModelFile)  
    with open(finalModelFile, 'w'):
        pass
    os.remove(predictFile)  
    with open(predictFile, 'w'):
        pass


def rle_mask(imgPath):
    def demask(mask): 
        s = mask.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T   
    img_masks = trueMasks.loc[trueMasks['ImageId'] == imgPath.replace(trainPath,''), 'EncodedPixels'].tolist()
    img = cv2.imread(imgPath)
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        all_masks += demask(mask)
    return all_masks  

def output(array):
    maxA = array.max()
    minA = array.min() 
    gradThrLo =  (minA/maxA)  
    gradThrHi =  (maxA-minA)/maxA           
    return (gradThrHi, gradThrLo)      

def store(model, imgName):
    entry = pd.DataFrame(np.array([[imgName,model[0],model[1]]],
                                  columns=['imgID', gradThrHi, gradThrLo] ) 
    entry.to_csv(recordFile, mode='a', header=False) 

def storePredict(imgName, IoU):
    IoU = 1 if IoU > iouThrh else IoU = 0   
    entry = pd.DataFrame(np.array([[imgName,IoU]]),
                                  columns=[ 'imgID', IoU ] ) 
    entry.to_csv( predictFile, mode='a', header=False) 

def finalModel():
    df = pd.read_csv(recordFile) 
    gradThrHi = df['gradThrHi'].median()
    gradThrLo = df['gradThrLo'].min() 
    entry = pd.DataFrame(np.array([[imgName,model[0],model[1]]],
                                  columns=['Final', gradThrHi, gradThrLo] ) 
    entry.to_csv(finelModelFile, mode='a', header=False) 
    print entry[0]  

def readModel():
    df = pd.read_csv(finalModelFile) 
    model = ( df.loc['final', 'gradThrHi'], df.loc['final', 'gradThrLo'] )  

def evalue(x, y):
    # IoU metric  
    I = np.multipy(x, y) 
    U = x + y 
    if U = 0:
        return 1  
    IoU = len(np.nonzero(I))/len(np.nonzero(U))    
    return IoU 
        
def calAccu():
    df = pd.read_csv( predictFile ) 
    acc = df[''].sum() / df[''].count() 
    return acc  

 
def test(fileList, model):  
    # test every img  
    for file in fileList:  
        # get predict
        predict = seg(file, model) 
        
        # get ground truth  
        trueMask = rle_mask(file)  

        # evaluate predict  
        evaluate = evalue( predict, trueMask ) 

        # store img,evaluate   
        storePredict(file.replace(trainPath,''), evaluate)        

    # compute accuracy   
    accuracy = calAccu() 
    print accuracy 

    return accuracy    
    
if __name__ == '__main__':
    model = readModel()      
    prediction = test(fileList, model) 



 
    
