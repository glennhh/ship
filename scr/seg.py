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

curPath = os.path.dirname(os.path.realpath(__file__))
trainPath = curPath + '/input/train/'
fileList = glob.glob(trainPath + "*.jpg")
imgsOri = io.ImageCollection(fileList)

#############################
###    Constants          ###
#############################
gradThrLo = 0.1 
gradThrHi = 0.6 
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

for file in fileList:
    imgOri = cv2.imread(file, cv2.IMREAD_UNCHANGED)   # read img

    img = cv2.GaussianBlur(imgOri,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # go gray 

    # find elevation  
    #elevation_map = sobel(img)
    gradient = rank.gradient(img, disk(2))  

    # find markers 
    markers = np.zeros_like(img)
    markers[img < np.max(gradient)*gradThrLo] = 1
    markers[img > np.max(gradient)*gradThrHi] = 2

    # fill regions with watershed transform    
    segmentation = watershed(gradient, markers)   # 1,2 
     
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
    
    
    
    
