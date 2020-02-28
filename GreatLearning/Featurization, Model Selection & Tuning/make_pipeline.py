# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:05:00 2020

@author: Niket
"""

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state = 0)

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(MinMaxScaler(), SVC())
print("Pipeline Steps: {}".format(pipe.steps))

pipe.fit(X_train, y_train)

print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))
