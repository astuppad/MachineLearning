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

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

"""#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""