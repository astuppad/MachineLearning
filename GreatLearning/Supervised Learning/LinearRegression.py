# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 08:18:03 2019

@author: N827941
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split

cData = pd.read_csv('auto-mpg.csv')
cData.shape

# 8 variables: 
#
# MPG (miles per gallon), 
# cylinders, 
# engine displacement (cu. inches), 
# horsepower,
# vehicle weight (lbs.), 
# time to accelerate from O to 60 mph (sec.),
# model year (modulo 100), and 
# origin of car (1. American, 2. European,3. Japanese).
#
# Also provided are the car labels (types) 
# Missing data values are marked by series of question marks.

# Droping the car_name column
cData = cData.drop('car name', axis = 1)

# Also replacing the categorical var with actual values
cData['origin'].replace({1 : 'america', 2: 'europe', 3: 'asia'}, inplace = True)
cData.head()

# Create Dummy Variables
cData = pd.get_dummies(cData, columns = ['origin'])
cData.head()

cData.describe()
cData.dtypes
#mpg               float64
#cylinders           int64
#displacement      float64
#horsepower         object
#weight              int64
#acceleration      float64
#model year          int64
#origin_america      uint8
#origin_asia         uint8

# isdigit()? on 'horsepower' 
hpIsdigit = pd.DataFrame(cData.horsepower.str.isdigit())

cData[hpIsdigit['horsepower'] == False]['horsepower']

# Missing values have a'?''
# Replace missing values with NaN
cData = cData.replace('?', np.nan)
cData.horsepower.median()

# replace the missing values with median value.
# Note, we do not need to specify the column names below
# every column's missing value is replaced with that column's median respectively  (axis =0 means columnwise)
#cData = cData.fillna(cData.median())
medianFiller = lambda x: x.fillna(x.median()) 
cData = cData.apply(medianFiller, axis = 0)
cData['horsepower'] = cData['horsepower'].astype('float64')

# BiVariate Plots
cData_attr = cData.iloc[:, 0:7]
sns.pairplot(cData, diag_kind='kde')

# lets build our linear model
# independant variables
X = cData.drop(['mpg','origin_europe'], axis = 1)
y = cData['mpg']

# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

# Fit Linear Model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)      

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))

intercept = regression_model.intercept_
print("The intercept for out model is {}".format(intercept))

regression_model.score(X_train, y_train)

regression_model.score(X_test, y_test)

# Adding interaction terms
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree = 2, interaction_only = True)
X_train2 = poly.fit_transform(X_train)
X_test2 = poly.fit_transform(X_test)

poly_clf = linear_model.LinearRegression()
poly_clf.fit(X_train2, y_train)

y_pred = poly_clf.predict(X_test2)

#In sample (training) R^2 will always improve with the number of variables!
print(poly_clf.score(X_train2, y_train))

#Out off sample (testing) R^2 is our measure of sucess and does improve
print(poly_clf.score(X_test2, y_test))

# but this improves as the cost of 29 extra variables!
print(X_train2.shape)
print(X_test2.shape)
