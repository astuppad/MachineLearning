# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 09:50:09 2019

@author: N827941
"""

import pandas as pd
import numpy as np

car_mpg = pd.read_csv('car-mpg.csv')

car_mpg = car_mpg.drop('car_name', axis = 1)
car_mpg['origin'] = car_mpg['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
car_mpg = pd.get_dummies(car_mpg, columns = ['origin'])
car_mpg = car_mpg.replace('?', np.nan)
car_mpg = car_mpg.apply(lambda x: x.fillna(x.median()), axis = 0)

X = car_mpg.drop('mpg', axis = 1)
y = car_mpg[['mpg']]

from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = X.columns

y_scaled = preprocessing.scale(y)
y_scaled = pd.DataFrame(y_scaled)
y_scaled.columns = y.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

for idx, column in enumerate(X_scaled.columns):
    print('The Coeficient for {0} is {1: 0.2f}'.format(column, regressor.coef_[0][idx]))
(regressor.score(X_train, y_train),regressor.score(X_test, y_test))

from sklearn.linear_model import Ridge
ridge = Ridge(alpha = 0.2)
ridge.fit(X_train, y_train)

for idx, column in enumerate(X_scaled.columns):
    print('The Coeficient for {0} is {1: 0.2f}'.format(column, ridge.coef_[0][idx]))
(ridge.score(X_train, y_train),ridge.score(X_test, y_test))

from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 0.3)
lasso.fit(X_train, y_train)


for idx, column in enumerate(X_scaled.columns):
    print('The Coeficient for {0} is {1: 0.2f}'.format(column, lasso.coef_[0][idx]))
(lasso.score(X_train, y_train),lasso.score(X_test, y_test))

# Let us generate polynomial models reflecting the non-linear interaction between some dimensions
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 1 , interaction_only = True)

X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_scaled)

# now fit the regression model and check the score
regressor.fit(X_train, y_train)
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)

# fit the ridge regression model and test the score
ridge = Ridge(alpha = 0.3, fit_intercept=True)
ridge.fit(X_train, y_train)
(ridge.score(X_train, y_train),ridge.score(X_test, y_test))


#fit the lasso regression model and check the score
lasso = Lasso(alpha=0.1, fit_intercept = True)
lasso.fit(X_train, y_train)
lasso.coef_
(lasso.score(X_train, y_train),lasso.score(X_test, y_test))


