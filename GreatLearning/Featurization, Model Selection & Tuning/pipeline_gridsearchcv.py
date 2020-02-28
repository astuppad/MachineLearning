# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 09:49:34 2020

@author: Niket
"""

import pandas as pd
df = pd.read_csv('wisc_bc_data.csv')
df.head()

X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values
print(X.shape)
print(y.shape)
print(df.info())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

le.transform(['M', 'B'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline
pipe_lr = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', LogisticRegression(random_state=1))])
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

# GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl', StandardScaler()), ('pca', PCA()), ('svc', SVC())])
param_grid = {'pca__n_components': [14, 15], 'svc__C': [0.001, 0.01, 0.1, 1, 10, 100], 'svc__gamma': [0.001, 0.01, 0.1, 1, 10], 'svc__kernel': ['rbf', 'poly']}
grid = GridSearchCV(pipe_svc, param_grid, cv = 5)
grid.fit(X_train, y_train)
print(' Best cross validation accuracy {:.2f}'.format(grid.best_score_))
print(' Best parameters', grid.best_params_)
print(' Test set accuracy {:.2f}'.format( grid.score(X_test,y_test)))

grid.predict(X_test)
