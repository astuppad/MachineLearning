# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 19:33:52 2019

@author: Niket
"""

import numpy as np #mathematics library
import matplotlib.pyplot as plt #plot chart 
import pandas as pd #import dataset and manage dataset

#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3],)
X[:, 1:3] = imputer.transform(X[:, 1:3])
