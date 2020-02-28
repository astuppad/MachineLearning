# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:13:21 2020

@author: Niket
"""

import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X, y = digits.data, digits.target

clf = RandomForestClassifier(n_estimators=50)

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["entropy", "gini"]}

samples = 10
randomCV = RandomizedSearchCV(clf, param_distributions= param_dist, n_iter = samples)

randomCV.fit(X, y)
print(randomCV.best_params_)

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["entropy", "gini"]}

grid_search = GridSearchCV(clf, param_grid= param_grid)

start = time()
grid_search.fit(X, y)

grid_search.best_params_
grid_search.cv_results_["mean_test_score"]

grid_search.best_estimator_
