# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:11:42 2020

@author: Niket
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

pima_df = pd.read_csv('pima-indians-diabetes.csv')

X = pima_df.drop('class', axis = 1)
y = pima_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression())])

pipeline.fit(X_train, y_train)

from sklearn import metrics

y_predict = pipeline.predict(X_test)
model_score = pipeline.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))
