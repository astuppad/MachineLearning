# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:43:38 2019

@author: N827941
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy, scipy.stats
import math

df = pd.read_csv('LGD.csv')
df.info()

sns.distplot(df['Losses in Thousands'], kde=False, bins = 50)
sns.distplot(df['Age'], kde = False, bins = 50)
sns.distplot(df['Years of Experience'])
sns.boxplot(x = 'Married', y = 'Losses in Thousands', data = df, hue = 'Gender')

# get the dummy variables
dummy_var1 = pd.get_dummies(df['Married'], drop_first = True)
dummy_var2 = pd.get_dummies(df['Gender'], drop_first = True)
df_new = pd.concat([df, dummy_var1, dummy_var2], axis = 1)
df_new2 = df.drop(['Gender', 'Married'], axis = 1)

X = df_new[['Age', 'Number of Vehicles', 'M', 'Single']]
y = df_new['Losses in Thousands']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

from sklearn.linear_model import LinearRegression
ln = LinearRegression()
ln.fit(X_train, y_train)
print(ln.coef_)
print(ln.intercept_)

pred = ln.predict(X_test)
from sklearn import metrics
print(metrics.mean_absolute_error(y_test, pred))
from sklearn.metrics import r2_score
r2_score(y_test, pred)

from statsmodels.api import add_constant
X2 = add_constant(X_train)
lm = sm.OLS(y_train, X2)
lm2 = lm.fit()
lm2.pvalues

regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
regression_model.score(X_train, y_train)