# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 11:05:08 2019

@author: N827941
"""
# Import Libraries and load the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes = pd.read_csv('Diabetes.csv')
print(diabetes.columns)

diabetes.head()

# Check dimension of dataset
print('dimension of diabetes data: {}'.format(diabetes.shape))

# Check distribution of dependent variable, Outcome and plot it
print(diabetes.groupby('Outcome').size())

#  Out of 768 data points, 500 are labeled as 0 and 268 as 1.
import seaborn as sns

sns.countplot(diabetes['Outcome'], label = 'Count')

diabetes.info()

# Check data distribution using summary statistics and provide your findings(Insights)
diabetes.describe().transpose()

# Do correlation analysis and bivariate viualization with Insights
colormap = plt.cm.viridis   
plt.figure(figsize = (15,15))
plt.title('Pearson Correlation of attributes', y = 1.05, size = 19)
sns.heatmap(diabetes.corr(), linewidths= 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = 'white', annot = True)

# Plot a scatter Matrix
spd = pd.scatter_matrix(diabetes, figsize = (20, 20), diagonal = "kde")

# Do train and test split with stratify sampling on Outcome variable to maintain the distribution of dependent variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != -1], diabetes['Outcome'], stratify = diabetes['Outcome'], random_state = 1)
X_train.shape

#  Train Support Vector Machine Model
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

print('Accuracy on training set: {: .2f}'.format(classifier.score(X_train, y_train)))
print('Accuracy on test set: {: .2f}'.format(classifier.score(X_test, y_test)))

# Scale the data points using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Fit SVM Model on Scale data and give your observation
svc = SVC()
svc.fit(X_train_scaled, y_train)
print('Accuracy on training set: {: .2f}'.format(svc.score(X_train_scaled, y_train)))
print('Accuracy on test set: {: .2f}'.format(svc.score(X_test_scaled, y_test)))

# Try improving the model accuracy using C=1000
svc = SVC(C = 1000)
svc.fit(X_train_scaled, y_train)

print('Accuracy on trianing set: {: .2f}'.format(svc.score(X_train_scaled, y_train)))
print('Accuracy on test set: {: .2f}'.format(svc.score(X_test_scaled, y_test)))