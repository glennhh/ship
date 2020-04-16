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


curPath = os.path.dirname(os.path.realpath(__file__))  
outputPath = curPath + '/output/' 
recordFile = outputPath + 'record.cvs'
finalModelFile = outputPath + 'model.cvs'     
trainPath = curPath + '/input/train/'
fileList = glob.glob(trainPath + "*.jpg")
imgsOri = io.ImageCollection(fileList)
trueMasks = pd.read_csv(curPath + '/input/train_ship_segmentations_v2.csv')  
trueMasks.head()
shape=(768, 768)  

#############################
###    hyperparemeter     ###
#############################
hypA   =   1.6
patchHoleSize = 20


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
    try: 
        os.remove(recordFile) 
    except FileNotFoundError:
        pass
    with open(recordFile, 'w'): 
        entry = pd.DataFrame([], columns= ['imgName', 'gradThrHi', 'gradThrLo'] )  
        entry.to_csv(recordFile) 
    try: 
        os.remove(finalModelFile) 
    except FileNotFoundError:
        pass  
    with open(finalModelFile, 'w'):
        pass

def rle_mask(imgPath):
    def demask(mask): 
        #print( type(mask),mask )  
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
    #print(max(img_masks)) 
    img = cv2.imread(imgPath)
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        all_masks += demask(mask)
    return all_masks  

def output(array):
    #print(len(array), len(array[0]))
    maxA = array.max()
    minA = array.min()
    meanA = array.mean() 
    gradThrLo =  (minA*2/maxA)  
    #gradThrHi =  (maxA-minA)/maxA           
    gradThrHi = meanA * hypA / maxA   
    return (gradThrHi, gradThrLo)      

def store(model, imgName):
    entry = pd.DataFrame( { 'imgName': [imgName], 'gradThrHi': [model[0]], 'gradThrLo': [model[1]] }, 
                          columns= ['imgName', 'gradThrHi', 'gradThrLo'] )  
    #entry = pd.DataFrame( [ imgName, model[0], model[1] ] ) 
    #entry.append(pd.read_csv(recordFile, header=None), ignore_index=True)    
    entry.to_csv(recordFile, mode='a', header=False, index=True) 

def finalModel():   
    df = pd.read_csv(recordFile) 
    gradThrHi = df[df.gradThrHi > 0].median().gradThrHi    # exclude 0 (no ship)  
    gradThrLo = df[df.gradThrLo > 0].median().gradThrLo 
    entry = pd.DataFrame( {'gradThrHi': [gradThrHi], 'gradThrLo': [gradThrLo] }, columns = ['gradThrHi', 'gradThrLo'] )    # save to csv   
    entry.to_csv(finalModelFile, mode='a', header=True) 
    print('\nModel: \n\n', entry, '\n')
 
def train():  
    prep()  
    # initialize model
    #gradThrLo = 0.1 
    #gradThrHi = 0.6    # 0.6 is best    
    #model = (gradThrLo, gradThrHi)  

    # start training
    fileAm = len(fileList) 
    print('Start training ......')   
    print('Total : %d' % fileAm ) 
    for file in fileList:  
        # print progress     
        progress = int((fileList.index(file)+1)/fileAm*100)   
        if not int(progress%10):    
            print( '\t%.0f%% ' % progress ) 
        imgName = file.replace(trainPath,'') 
        imgOri = cv2.imread(file, cv2.IMREAD_UNCHANGED)   # read img

        # get decoded mask: y  
        trueMask = rle_mask(file) 

        # GaussianBlur  
        img = cv2.GaussianBlur(imgOri,(3,3),0)            # de-noise  

        # Gray 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # go gray 
 
        ##img = measure.block_reduce( img, (2,2), np.mean ) 
        ##img = np.array( img, dtype=np.uint8 ) 
        ##img = ndi.zoom(img, 2, order=0) 
        ##img = ndi.zoom(img, 2, order=0) 
 
        # gradient   
        gradient = rank.gradient(img, disk(3))  
         
        # if gt is 0 
        if trueMask.max() == 0:
            model = (0,0)  
            store( model, imgName )  
            continue  
                         
        # filter grad in mask 
        maskGrad = np.multiply( trueMask, gradient )   
        pureMaskGrad = maskGrad[np.nonzero(maskGrad)]    
        #print( np.sort(pureMaskGrad)[int(len(list(pureMaskGrad))*0.89) ]) 
        # get model     
        model = output(pureMaskGrad)  

        # store model  
        store(model, imgName)  

       
        ''' 
        # find markers 
        markers = np.zeros_like(img)  
        markers[img < np.max(gradient)*gradThrLo] = 1  
        markers[img > np.max(gradient)*gradThrHi] = 2  
        #show2( markers, 'markers', trueMask, 'trueMask')    
        #sys.exit() 
 
        # fill regions with watershed transform    
        segmentation = watershed(gradient, markers)   # 1,2 
        #print(gradient[410], markers[410]) 
        #show2( markers, 'markers', gradient, 'gradient')    
        #sys.exit()   
 
        # patch holes 10001  
        segmentation = ndi.binary_fill_holes(segmentation - 1, structure=np.ones((patchHoleSize,1)))  # 0,1 
        labeledShip, _ = ndi.label(segmentation)       # 0,3  
    
        # threshold image
        segmentation = segmentation.astype(np.uint8)*255
        imgMask = np.zeros(segmentation.shape,np.uint8)      # Image to draw the contours
        contours,hierarchy = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imgMask,[cnt],0,(0,0,255),2)   # draw contours in green color
            cv2.polylines(imgMask,[box],True,(255,0,0),2)   # draw rectangle in blue color
        # fill holes in mask
        imgMask = ndi.binary_fill_holes(imgMask)
        show2(imgOri, 'Input', imgMask, 'Mask')
        '''
    # save final model 
    finalModel()     

    return model  
    
if __name__ == '__main__':
    model = train() 


 
    
