# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:39:16 2019

@author: N827941
"""
# import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# import the dataset 
dataset = pd.read_excel('Dataset.xlsx')

# check whether the independent variables are object so that we can determine whether are there any missing values
dataset.dtypes

dataset.qcode.value_counts()
# qcode 5 questions

# define X and y variable so the X should be 
X = dataset.iloc[:, [2]].values
y = pd.DataFrame(dataset.R)

# split the data into train and test with 70% training data and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)

# train the data with Gaussian Naive bayes algorithm
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# now predict with test data
y_pred = classifier.predict(X_test)

# measure the perfomance with test data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

