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
#trainPath = '/home/hao/Data/course/ML/project/train_v2/'
imgNameFile = curPath + '/input/imgList.csv'
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
    try:
        os.remove(predictFile)
    except FileNotFoundError:
        pass
    with open(predictFile, 'w'):  
        entry = pd.DataFrame([], columns= ['imgName', 'IoU'] )
        entry.to_csv(predictFile)    

def rle_mask(imgPath):
    def demask(mask): 
        try:
            s = mask.split()
        except AttributeError:
            return np.zeros((768, 768))
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
                                  columns=['imgID', gradThrHi, gradThrLo] ))  
    entry.to_csv(recordFile, mode='a', header=False) 

def storePredict(imgName, IoU):
    #IoU = 1 if IoU > iouThrh else IoU = 0   
    entry = pd.DataFrame( { 'imgName': [imgName], 'IoU': [IoU] }, columns = ['imgName', 'IoU'] )   
    entry.to_csv( predictFile, mode='a', header=False) 

def finalModel():
    df = pd.read_csv(recordFile) 
    gradThrHi = df['gradThrHi'].median()
    gradThrLo = df['gradThrLo'].min() 
    entry = pd.DataFrame(np.array([[imgName,model[0],model[1]]],
                                  columns=['Final', gradThrHi, gradThrLo] ))   
    entry.to_csv(finelModelFile, mode='a', header=False) 
    print(entry[0] ) 

def readModel():
    df = pd.read_csv(finalModelFile) 
    model = ( df['gradThrHi'][0] , df['gradThrLo'][0]  ) 
    print('Current model is: \t ', model) 

def evalue(x, y):
    # IoU metric  
    I = np.multiply(x, y) 
    U = x + y 
    if np.max(U) == 0:
        return 1  
    IoU = len(np.nonzero(I))/len(np.nonzero(U))    
    return 1 if IoU > iouThrh else 0     
        
def calAccu():
    df = pd.read_csv( predictFile ) 
    a = df[df.IoU > 0].count().IoU
    b = df[df.IoU >= 0].count().IoU
    acc = a / b  
    print('\n\tCorrect:     ', a, '\n\tTotal:       ', b, '\n\n\tAccuracy:    ', acc, '\n')    
    return acc  

def test(fileList, model):  
    
    # prepare 
    prep() 
    
    fileAm = len(fileList) 
    print('Start testing ......')   
    print('Total : %d' % fileAm ) 
    i, j = 0, list(map(int, np.multiply( fileAm, np.multiply(0.1, range(1,11,1)) )))

    # test every img  
    for file in fileList:  
        file = trainPath + file

        # print progress     
        i = i + 1
        if i in j:
            print( '\t{:.0%}'.format((j.index(i)+1)/10 ))

        # get predict
        predict = seg(file, model) 
        
        # get ground truth  
        trueMask = rle_mask(file)  
        
        # evaluate predict  
        evaluate = evalue( predict, trueMask ) 
        show2(trueMask, 'true', predict, str(evaluate))    

        # store img,evaluate   
        storePredict(file.replace(trainPath,''), evaluate)        

    # compute accuracy   
    accuracy = calAccu() 

    return accuracy    
    
if __name__ == '__main__':
    model = readModel()      

    imgFile = open(imgNameFile,'r').readlines()
    fileList = [ value[:-1] for value in imgFile ]   # remove \n
    
    accuracy = test(fileList, model) 



 
    
