#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 02 20:54:21 2019

@author: varshaganesh
"""
import cv2
import argparse
import numpy as np
from ViolaJonesFaceDetector import FaceDetector
from utils import nonMaximalSupression,getImages, saveJson

"""
     Declaring Global Variables
     
"""

JSON_LIST = []

def addToJson(filename,locations):
    """
        Adding the detected Faces in an Image to the JSON_LIST 
     
    """   
    
    for i in range(0,len(locations)):  
        
        x1,y1,x2,y2 = locations[i]
        
        element = {"iname": filename, "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)]} 
        
        JSON_LIST.append(element)

    
def parse_args():   
    
    """ 
        Function to Parse the Command Line Argument which specifies the Data Directory that contains the Test Data
    """   
    parser = argparse.ArgumentParser(description="CSE 473/573 project 3.")
    
    parser.add_argument('string', type=str, default="./data/",help="Resources folder,i.e, folder in which Test images are stored") 
    
    args = parser.parse_args()   
    
    return args  

 

def getFaceLocations(gray):
    
    rowssize = 50
            
    colssize = 50
                                    
    clf = FaceDetector.Load('Classifier')
            
    locations = []
            
    while rowssize<(len(gray)-2):
                
        for r in range(0,gray.shape[0] - rowssize, 10):
                    
            for c in range(0,gray.shape[1] - colssize, 10):
                        
                window = gray[r:r+rowssize,c:c+colssize]
                        
                window=cv2.resize(window,dsize=(24,24))
                        
                prediction = clf.classify(window)
                        
                if prediction ==1:
                                        
                    locations.append([r,c,r+rowssize,c+colssize])
                    
        colssize+=50
        
        rowssize+=50
    
    return locations
    
                
    


if __name__ == '__main__':
    
    """
        The main Function Does the following:
            
        1. Takes in an command-line argument as input, that contains the Test Images
        
        2. For every Image
        
            2.1 Traverses the image to detect the faces in the image using Viola Jones Classifier
            
            2.2 Gets the locations in which the image has Faces
            
            2.3 Adds the locations in the global JSON_LIST
            
        3. Dumps the JSON_LIST to results.json
        
        4. STOP        
    """
    args = parse_args()
    
    TestImgsFolder = args.string
    
    TestImgs = getImages(TestImgsFolder)
    
    for i in range(0, len(TestImgs)):
        
        imgtest = cv2.imread(TestImgsFolder+'/'+ TestImgs[i])
                    
        gray = cv2.cvtColor(imgtest,cv2.COLOR_RGB2GRAY)

        locations = getFaceLocations(gray)
        
        if locations is not None:
                
            locations = nonMaximalSupression(np.array(locations), 0.25)
                            
            addToJson(TestImgs[i],locations)

    saveJson(JSON_LIST)
    

    
    
    
    
        
    
    
    

    
