# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:15:25 2019

@author: N827941
"""
# We will use kNN to predict the type of Breast Cancer in the Breast Cancer Wisconsin(Diagnostic)Data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data
bcData = pd.read_csv('wisc_bc_data.csv')
bcData.shape
bcData.dtypes

bcData['diagnosis'] = bcData.diagnosis.astype('category')
bcData.describe().transpose()

bcData.groupby(['diagnosis']).count()

# drop the first column from the data frame. This is Id column which is not used in modeling
# The first column is id column which is patient id and nothing to do with the model attriibutes. So drop it.
X = bcData.drop(labels = 'diagnosis', axis = 1)
y = bcData['diagnosis']

# convert the features into z scores as we do not know what units / scales were used and store them in new dataframe
# It is always adviced to scale numeric attributes in models that calculate distances.
XScaled = X.apply(zscore)

XScaled.describe()

# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Build the KNN 
KNN = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')

KNN.fit(X_train, y_train)

# Evaluate the performance of the model
# For every test data point, predict it's label based on 5 nearest neighbours in this model. The majority class will 
# be assigned to the test data point
predicted_labels = KNN.predict(X_test)
KNN.score(X_test, y_test)

from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predicted_labels, labels = ["M", "B"])

df_cm = pd.DataFrame(cm, index = [i for i in ["M", "B"]],
                         columns = [i for i in ["Predicted M", "Predicted B"]])

plt.figure(figsize = (7, 5))
sns.heatmap(df_cm, annot = True)


# Choosing a k value
scores = []
for k in range(1, 50):
    classifier = KNeighborsClassifier(n_neighbors = k, weights = 'distance')
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))

plt.plot(range(1, 50), scores)

