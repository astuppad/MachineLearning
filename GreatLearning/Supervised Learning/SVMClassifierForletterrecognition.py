# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:03:53 2019

@author: N827941
"""
# Import the libraries
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# function to calculate the accuracies
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if(testSet[x] == predictions[x]):
            correct += 1
    return (correct/float(len(testSet))) * 100

# Read the data
lData = pd.read_csv('letterdata.csv')

# Split the X, y coordinates
X, y = np.array(lData)[:, 1:16], np.array(lData.letter)[:]

# Split the data
X_train = X[:16000, :]
X_test = X[16000:, :]
y_train = y[:16000]
y_test = y[16000:]

# Build a model
classifier = svm.SVC(gamma = 0.025, C = 3)
#gamma is a measure of influence of a data point. It is inverse of distance of influence. C is penalty of wrong classifications.

# fit the data in the classifier
classifier.fit(X_train, y_train)

# now predict the test data
y_pred = classifier.predict(X_test)

# calculate the accuracy with the function that is written earlier
getAccuracy(y_test, y_pred)

y_grid = (np.column_stack([y_test, y_pred]))

np.savetxt('ocr.csv', y_grid, fmt = '%s')

import string
lab = list(string.ascii_uppercase[0:26])
plab = ["Pr " + s for s in lab]

# Filter those cases where the model committed mistake and analyze the mistake, which characters most mistakes occured on?
# import the metrics and seaborn libraries to plot the heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred, labels = lab)

print(cm)

from sklearn import metrics
import seaborn as sns

cm=metrics.confusion_matrix(y_test, y_pred, labels=lab)

df_cm = pd.DataFrame(cm, index = [i for i in lab],
                  columns = [i for i in plab])
plt.figure(figsize = (20,16))
sns.heatmap(df_cm, annot=True ,fmt='g')