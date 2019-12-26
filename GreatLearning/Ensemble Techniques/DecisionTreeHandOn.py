# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:29:44 2019

@author: N827941
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('pima-indians-diabetes.csv')

ds.head(10)

ds.info()

ds.shape

X = ds.drop('class', axis = 1)
y = ds['class']
X = X.apply(lambda x: x.replace(0, x.mean()))
X.head(10)

X.describe()

sns.distplot(y)

sns.pairplot(ds)

ds.corr()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn import tree
dt = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 100, max_depth = 8, min_samples_leaf = 5)
tree.plot_tree(dt.fit(X_train, y_train))

from sklearn.metrics import accuracy_score, confusion_matrix
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index = ["0", "1"], columns = ["Predicted 0", "Predicted 1"])
sns.heatmap(cm_df, annot = True)