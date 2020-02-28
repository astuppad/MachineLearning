# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:28:51 2020

@author: Niket
"""

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 1)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()

knn_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
param_grid = { 'n_neighbors': list(range(1, 9)),
              'algorithm': ('auto','ball_tree', 'kd_tree', 'brute')}

from sklearn.model_selection import  GridSearchCV
gs = GridSearchCV(knn_clf, param_grid)

gs.fit(X_train, y_train)

gs.best_params_

gs.cv_results_['params']

gs.cv_results_['mean_test_score']