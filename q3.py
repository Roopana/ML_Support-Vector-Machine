#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:44:37 2019

@author: roopana
"""
from sklearn import datasets
from crossValidation import mycrossval
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def loadDataSets():        
    boston = datasets.load_boston()
    
    ## Creating Boston50 dataset
    dfBoston50 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston50['response'] = boston['target']
    dfBoston50['target'] = dfBoston50['response']>= np.percentile(dfBoston50['response'],50)
    dfBoston50['target'] = dfBoston50['target']*1
    dfBoston50.loc[dfBoston50.target==0, 'target'] = -1

    # Creating Boston75 dataset
    dfBoston75 = pd.DataFrame(data=boston['data'], columns = boston['feature_names'])
    dfBoston75['response'] = boston['target']
    dfBoston75['target'] = dfBoston75['response']>= np.percentile(dfBoston75['response'],75)
    dfBoston75['target'] = dfBoston75['target']*1
    dfBoston75.loc[dfBoston75.target==0, 'target'] = -1
    #Construct dfDigits Data
    digits = datasets.load_digits(n_class = 10, return_X_y = False)
    dfDigits = pd.DataFrame(data = digits.data)
    dfDigits['target'] = digits.target
    
    return dfBoston50, dfBoston75, dfDigits

#Normalises data by subtracting mean and dividing by Stand deviation
def getNormData(x):
    normd = x
    for i in range(x.shape[1]):
    	mean = np.mean(x.iloc[:,i])
    	std = np.std(x.iloc[:,i])
    	normd.iloc[:,i] =  np.divide(np.subtract(x.iloc[:,i], mean), std)
    return normd

# calls cross validationa and thus executing classification
def run():
    dfBoston50, dfBoston75, dfDigits = loadDataSets()

    x50 = dfBoston50.iloc[:, 0:13]
    y50 = dfBoston50.iloc[:, 14] # Take the target column as y and not the actualresponse column
    x75 = dfBoston75.iloc[:, 0:13]
    y75 = dfBoston75.iloc[:, 14]
    #xDigits = dfDigits.iloc[:, 0:64]
    #yDigits = dfDigits.iloc[:, 64]

    x50 = getNormData(x50)
    x75 = getNormData(x75)

    k=5 # No of folds for Cross Validation
    nDim = 13 # no of dimensions in the input dataset
    method = 'mySVM2'
    
    print("Applying mySVM2 on Boston 50 Data")
    mycrossval(method, x50, y50, k, nDim)
    
    print("Applying mySVM2 on Boston 75 Data")
    mycrossval(method, x75, y75, k, nDim)
    
    method = 'LogisticRegression'
   
    print("Applying LogisticRegression on Boston 50 Data")
    mycrossval(method, x50, y50, k, nDim)

    print("Applying LogisticRegression on Boston 75 Data")
    mycrossval(method, x75, y75, k, nDim)

run()
