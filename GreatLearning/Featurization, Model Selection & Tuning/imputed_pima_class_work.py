# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 20:21:42 2020

@author: Niket
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier

pima_df = pd.read_csv('pima-indians-diabetes.csv')

pima_df[['Preg', 'Plas', 'Pres', 'skin', 'test', 'mass', 'pedi', 'age']] = pima_df[['Preg', 'Plas', 'Pres', 'skin', 'test', 'mass', 'pedi', 'age']].replace(0, np.NaN)

pima_df.head()

import seaborn as sns

sns.pairplot(pima_df, hue='class', diag_kind='kde')

values = pima_df.values
X = values[:, 0:8]
y = values[:, 8]

imputer = Imputer()

imputer = Imputer(missing_values=np.NaN, strategy='median', axis = 0)


transformed_x = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(transformed_x, y, test_size = 0.3, random_state = 7)

dt_model = DecisionTreeClassifier(criterion='entropy')
dt_model.fit(X_train, y_train)

dt_model.score(X_test, y_test)

y_predict = dt_model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predict))

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth= 8)
dt_model.fit(X_train, y_train)

dt_model.score(X_test, y_test)

y_predict = dt_model.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predict))