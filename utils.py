#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 03 12:08:47 2019

@author: varshaganesh

A Lsit of Utility Methods to support Viola-Jones Face Detection Algorithm.

"""
import pickle
from ViolaJonesFaceDetector import FaceDetector
import numpy as np
import os
import json
import cv2
import random

def getImages(folder):
    
    """ 
        Function to Retreive the names of Img files with Extension .jpg from the desired Folder
    """
    
    images = []
    
    for filename in os.listdir(folder):
        
        if filename.endswith('.jpg'):
            
            images.append(filename)
            
    images.sort()
    
    return images




def saveJson(JSON_LIST):
    """
        Dumps the JSON_LIST to result.json
        
    """
    output_json = "results.json"
    
    with open(output_json, 'w') as f:
        
        json.dump(JSON_LIST, f)
        
        
def nonMaximalSupression(bounding_boxes, maxOverLap):
    
    """
        As we scale the features to match the Image and compare with different sizes, Multiple regions of a face may be detected by the classifier.
        
        Inorder to make sure the same face is not produced at several instances in the result, we have to suppress the result using Non Maximal Suppression Algorithm,as below.
        
        Reference: https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
        
    """
    if len(bounding_boxes) == 0: 
        return []
    
    pick = []
    
    x1 ,y1,x2, y2= bounding_boxes[:,0], bounding_boxes[:,1],bounding_boxes[:,2],bounding_boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    indexes = np.argsort(y2)
    
    while len(indexes) > 0:

        last = len(indexes) - 1
        
        i = indexes[last]
        
        pick.append(i)
        
        suppress = [last]
        
        for pos in range(0, last):
            
            j = indexes[pos]
 
            xx1 = max(x1[i], x1[j])
            
            yy1 = max(y1[i], y1[j])
            
            xx2 = min(x2[i], x2[j])
            
            yy2 = min(y2[i], y2[j])
            
            w = max(0, xx2 - xx1 + 1)
            
            h = max(0, yy2 - yy1 + 1)
 
            overlap = float(w * h) / area[j]
 
            if overlap > maxOverLap:
                
                suppress.append(pos)
 
        indexes = np.delete(indexes, suppress)
 
        return bounding_boxes[pick]
    
    

def TrainDetector(NoOfClassifiers):
    
    """ 
        Method to Initialize the Training Process
    """
    classifier = FaceDetector(NoOfClassifiers)
    
    with open('Training-Data.pkl', 'rb') as f:
        
        training = pickle.load(f)
        
    classifier.Train(training, 800, 800)
    
    classifier.save('Classifier')


def TrainData():
    
    """ Retreiving Positive and Negative images and storing them in a Pickle File
    
        Once this pickle file is generated we don't have to retreive images or convert them to gray scale again
        
        Until or unless you want to change the Training Data Set
    """
    
    Training_Data = []
    
    for filename in os.listdir("./PositiveData/"):
        
        if filename.endswith(".jpg"):
            
            img=cv2.imread('./PositiveData/'+filename)
            
            img = cv2.resize(img,dsize=(24,24))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            
            data = (img,1)
            
            Training_Data.append(data)
            
    for filename in os.listdir("./NegativeData/"):
        
        if filename.endswith(".jpg"):
            
            img=cv2.imread('./NegativeData/'+filename)
            
            img = cv2.resize(img,dsize=(24,24))
            
            img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            
            data = (img,0)
            
            Training_Data.append(data)
            
    random.shuffle(Training_Data)
        
    Training = open('./Training-Data.pkl','wb')
        
    pickle.dump(Training_Data,Training)
        
    Training.close()
        

    
 
