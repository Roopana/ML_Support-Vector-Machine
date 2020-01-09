#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:44:37 2019

Has implementation of 'MySVM2'

@author: vcroopana
"""

import numpy as np

class MySVM2:
    
    d = 0
    stepSize = 0
    w = np.zeros([1,d])
    lambd = 0
    wSum = 0
    ## consructor param d - no of attributes in given data
    def __init__(self, d):
        self.d = d
        self.stepSize = 0.001 #assume 
        self.w = np.random.uniform(-0.01, 0.01, [1,self.d])
        self.wSum = 0
        self.w0 = 0
        self.lambd = 5 # as per qn
    
    def calculateError(self, X, y):
        deltaE = 0
        deltaEw0 =0
        X = X.values
        y = y.values
        error =0
        # Calculate Error function value as per given loss function
        for i in range(X.shape[0]):
            wtX = np.dot(self.w, X[i].T)
            wtXPlusW0 = wtX + self.w0 # f(x)
            currError = np.maximum(0, 1-(np.dot(y[i], wtXPlusW0))) # Scalar 
            error = error + currError
            if (np.dot(y[i], wtXPlusW0))<1 : # y.f(x) <0 implies incorrect classification
                ## wt+1 ← wt − η(λwt − yixi)
                deltaE = deltaE + (self.lambd*self.w - np.dot(y[i],X[i]))
            else: ##  wt+1 ← wt − ηλwt
                deltaE = deltaE + self.lambd*self.w                 
            deltaEw0 = deltaEw0 + (wtXPlusW0 - y[i])
        deltaE = np.divide(deltaE, X.shape[0])
        error = np.divide(error, X.shape[0])+ (self.lambd*0.5*np.linalg.norm(self.w)*np.linalg.norm(self.w))

        return error, deltaE, deltaEw0
              
    def calculateW(self, X, y):

        error, deltaE, deltaEw0 = self.calculateError(X, y)
        prevE = 0
        #find W and W0 that give almost 0 change in error function using Gradient Descent approach
        while(abs(prevE-error)> 0.0001):  
            prevE = error
            # Update w0 and w class variables in each iteration
            self.w = np.subtract(self.w, self.stepSize * deltaE) 
            self.w0 = np.subtract(self.w0, self.stepSize * deltaEw0)
            # Recalculate error and delta E values using updated values of w and w0
            error, deltaE, deltaEw0 = self.calculateError(X, y)
        
    def fit(self, X,y):
        self.calculateW(X, y)
    
    def predict(self,X):
        predictedY = np.zeros([X.shape[0]])
        for row in range(X.shape[0]):
            svm = np.dot(self.w, X.values[row].T) + self.w0
            if svm>= 0: 
                predictedY[row] = 1
            else:
                predictedY[row] = -1
        return predictedY  


