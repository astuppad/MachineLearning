# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:24:10 2019

@author: N827941
"""
# Import libraries and load the dataset
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
import pandas as pd

df = pd.read_csv('iris-1.csv')
df.sample(10)

# Estimating missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
imputer.fit(df.iloc[:, :-1])
data_impute = imputer.transform(df.iloc[:, :-1].values)
df.iloc[:, :-1] = data_impute
iris = df

# Dealing with categorical data
iris.iloc[:, 5].unique()

from sklearn.preprocessing import LabelEncoder
lblencoder = LabelEncoder()
iris.iloc[:, -1] = lblencoder.fit_transform(iris.iloc[:, 5])

# Observe the association of each independent variable with target variable and drop variables from feature set having correlation in range -0.1 to 0.1 with target variable.
iris.corr()

# Observe the independent variables variance and drop such variables having no variance or almost zero variance(variance < 0.1). They will be having almost no influence on the classification
iris.var()

# Plot the scatter matrix for all the variables.
splt = pd.scatter_matrix(iris,  c = iris.iloc[:, :-1], figsize = (20, 20), marker = 'o')

# Split the dataset into training and test sets with 80-20 ratio
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(iris.ix[:, 1:5])
y = np.array(iris['Species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# learning (k=3)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

# learning (k=5)
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

# learning (k=9)
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

myList = list(range(1, 20))
neighbors = list(filter(lambda x: x%2 != 0, myList))
scores = []
for i in neighbors:
    knn = KNeighborsClassifier(n_neighbors = i, weights = 'distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# changing to misclassification error
MSE = [1 - x for x in scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]

import matplotlib.pyplot as plt

plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors k')
plt.ylabel('Misclassification error')