#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 07 20:24:42 2019

@author: varshaganesh

A Python implementation of the Viola-Jones Face Detection Algorithm.

References: 
    
    1. https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
    
    2. https://medium.com/datadriveninvestor/understanding-and-implementing-the-viola-jones-image-classification-algorithm-85621f7fe20b
    
    3. https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
    
    4. https://github.com/btuan/ViolaJones/blob/master/violajones.py

"""
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif


def getIntegralImage(img):
    
    """
        Function to Compute the integral image of a given Image using the formula: 
            
        Integral_Img[i][j] = img[i][j] + integral[i-1][j] + integral[i][j-1] + - integral[i-1][j-1]
        
        provided the index are valid, i.e, i-1 >=0 and j-1 >=0
    """
    row = len(img)
    
    col = len(img[0])
    
    integral = np.zeros((row,col))
    
    for i in range(0,row):
        
        for j in range(0,col):
            
            integral[i][j] = int(img[i][j])
            
            if i-1 >=0 and j-1 >=0:
                
                integral[i][j] = integral[i][j] + integral[i-1][j] + integral[i][j-1] + - integral[i-1][j-1] 
                
            elif i-1 >= 0:
                
                integral[i][j] = integral[i][j] + integral[i-1][j]
                
            elif j-1 >= 0:
                
                integral[i][j] = integral[i][j] + integral[i][j-1]
            
    return integral



        
class Box:
    
    def __init__(self, x, y, width, height):
        
        self.x = x
        
        self.y = y
        
        self.width = width
        
        self.height = height
    
    def compute_feature(self, integralImg):
        """
            Computes the value of the Box given the integral image
        """
        return integralImg[self.y+self.height][self.x+self.width] + integralImg[self.y][self.x] - (integralImg[self.y+self.height][self.x]+integralImg[self.y][self.x+self.width])


class Classifier:
    def __init__(self, positive, negative, threshold, polarity):
        """
            Initializes a Classifier
        """
        self.positive = positive
        
        self.negative = negative
        
        self.threshold = threshold
        
        self.polarity = polarity
    
    def classify(self, x):
        """
            Classifies an integral image based on a feature f and the classifiers threshold and polarity 
        """
        feature = lambda integralImg: sum([pos.compute_feature(integralImg) for pos in self.positive]) - sum([neg.compute_feature(integralImg) for neg in self.negative])
        
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    

class FaceDetector:
    
    def __init__(self, NoOfClassifiers = 20):
        """
            Initializes the Face Detector with NoOfClassifiers
            
            NoOfClassifiers: The number of classifiers which should be used
        """
        self.NoOfClassifiers = NoOfClassifiers
        
        self.alphas = []
        
        self.classifiers = []

    def Train(self, Training, NoOfPositives, NoOfNegatives):
        """
            Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
            
            Training: A List of tuples (Image, Classification (1 - if positive image 0 - if negative image) 
            
            NoOfPositives: the number of positive samples
            
            NoOfNegatives: the number of negative samples
            
        """
        print('Training Started.......')
        
        weights = np.zeros(len(Training))
        
        training_data = []
        
        for x in range(len(Training)):
            
            training_data.append((getIntegralImage(Training[x][0]), Training[x][1]))
            
            if Training[x][1] == 1:
                
                weights[x] = 1.0 / (2 * NoOfPositives)
                
            else:
                
                weights[x] = 1.0 / (2 * NoOfNegatives)

        features = self.BuildFeatures(training_data[0][0].shape)
        
        X, y = self.apply_features(features, training_data)
        
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        
        X = X[indices]
        
        features = features[indices]
        

        for t in range(self.NoOfClassifiers):
            
            weights = weights / np.linalg.norm(weights)
            
            weak_classifiers = self.TrainWeak(X, y, features, weights)
            
            clf, error, accuracy = self.SelectBest(weak_classifiers, weights, training_data)
            
            beta = error / (1.0 - error)
            
            for i in range(len(accuracy)):
                
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
                
            alpha = math.log(1.0/beta)
            
            self.alphas.append(alpha)
            
            self.classifiers.append(clf)
            
            print('Training Classifier : '+ str(t)+ 'Alpha : '+ str(alpha))
            
        print('success: Training Done')
            
        

    def TrainWeak(self, X, y, features, weights):
        """
            Finding the optimal thresholds for each classifier given the current weights
        """
        total_pos, total_neg = 0, 0
        
        for w, label in zip(weights, y):
            
            if label == 1:
                
                total_pos += w
                
            else:
                
                total_neg += w

        classifiers = []
                
        for index, feature in enumerate(X):
            
            print('Training feature '+ str(index) + 'out of ' + str(len(X)))
                            
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            
            pos_weights, neg_weights = 0, 0
            
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            
            for w, f, label in applied_feature:
                
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                
                if error < min_error:
                    
                    min_error = error
                    
                    best_feature = features[index]
                    
                    best_threshold = f
                    
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    
                    pos_seen += 1
                    
                    pos_weights += w
                    
                else:
                    neg_seen += 1
                    
                    neg_weights += w
            
            clf = Classifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            
            classifiers.append(clf)
            
        return classifiers
                
    def BuildFeatures(self, image_shape):
        """
            Builds the possible features given an image shape         
        """
        height, width = image_shape
        
        features = []
        
        print('Building features....')
        
        for w in range(1, width+1):
            
            for h in range(1, height+1):
                
                i = 0
                
                while i + w < width:
                    
                    j = 0
                    
                    while j + h < height:
                        
                        """
                            2 rectangle feature -Horizontally Adjacent
                        """
                        immediate = Box(i, j, w, h)
                        
                        right = Box(i+w, j, w, h)
                        
                        if i + 2 * w < width: 
                            
                            features.append(([right], [immediate]))
                        """
                            2 rectangle feature - Vertically Adjacent
                        """
                        bottom = Box(i, j+h, w, h)
                        
                        if j + 2 * h < height: 
                            
                            features.append(([immediate], [bottom]))
                            
                        right_2 = Box(i+2*w, j, w, h)
                        
                        """
                            3 rectangle feature - Horizontally Adjacent
                        """
                        if i + 3 * w < width: 
                            
                            features.append(([right], [right_2, immediate]))
                            
                        bottom_2 = Box(i, j+2*h, w, h)
                        
                        """
                            3 rectangle feature - Vertically Adjacent
                        """
                        if j + 3 * h < height: 
                            
                            features.append(([bottom], [bottom_2, immediate]))
                            
                        """
                            4 rectangle features 
                        """
                        bottom_right = Box(i+w, j+h, w, h)
                        
                        if i + 2 * w < width and j + 2 * h < height:
                            
                            features.append(([right, bottom], [immediate, bottom_right]))
                            
                        j += 1
                        
                    i += 1
                    
        return np.array(features)

    def SelectBest(self, classifiers, weights, training_data):
        """
            Selects the best  classifier for the given weights
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        
        print('selecting best classifier out of '+ str(len(classifiers)))

        for clf in classifiers:
                        
            error, accuracy = 0, []
            
            for data, w in zip(training_data, weights):
                
                correctness = abs(clf.classify(data[0]) - data[1])
                
                accuracy.append(correctness)
                
                error += w * correctness
                
            error = error / len(training_data)
            
            if error < best_error:
                
                best_clf, best_error, best_accuracy = clf, error, accuracy
                
        return best_clf, best_error, best_accuracy
    
    def apply_features(self, features, training_data):
        """
            Maps features onto the training dataset
        """
        X = np.zeros((len(features), len(training_data)))
        
        y = np.array(list(map(lambda data: data[1], training_data)))
        
        i = 0
        
        print('Applying features to Training set...')

        for positive_regions, negative_regions in features:
                        
            feature = lambda integralImg: sum([pos.compute_feature(integralImg) for pos in positive_regions]) - sum([neg.compute_feature(integralImg) for neg in negative_regions])
            
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            
            i += 1
            
        return X, y

    def classify(self, image):
        """
            Classifies an image
            Returns 1 if the image has a face and returns 0 otherwise
        """
        total = 0
        
        integralImg = getIntegralImage(image)
        
        for alpha, clf in zip(self.alphas, self.classifiers):
            
            total += alpha * clf.classify(integralImg)
            
        return 1 if total >= 0.6 * sum(self.alphas) else 0

    def save(self, filename):
        """
            Saves the classifier to a pickle
        """
        with open(filename+".pkl", 'wb') as f:
            
            pickle.dump(self, f)

    @staticmethod
    def Load(filename):
        """
            A static method which loads the classifier from a pickle
        """
        with open(filename+".pkl", 'rb') as f:
            
            return pickle.load(f)

