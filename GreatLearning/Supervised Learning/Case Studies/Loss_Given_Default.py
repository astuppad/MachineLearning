# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:43:38 2019

@author: N827941
"""

"""
Case Study on Loss Given Default

Context:
CNB Bank deals in all kinds of car loans. Customer first apply for loan after that company validates the customer eligibility for loan. In case the borrower doesn’t pay back the loan, the losses are to be incurred by the bank. LGD stands for Loss given default so it means when a customer at a bank defaults on his loan how much money does the bank lose. The customer might have paid some amount back or no amount at all.The bank wants to know if the amount the bank loses can be predicted for new customers who apply for a loan from the past data of all defaulters and their pending amounts

Problem:
The bank wants to automate the loss estimation based on customer detail provided while applying for loan. These details are Age, Years of Experience, Number of cars, Gender, Marital Status. To automate this process, they have given a problem to identify the loss estimation given that the customers is a defaulter, those are eligible for loan amount so that they get to know what features are leading to defaults up to which amount. Here are the details about the data set.

Data:
Variable - Description 
Ac_No - The account of customer used as identifier 
Age - Age of borrower (16-70) 
Years of Experience - Working experience (0-53) 
Number of Cars - Possessed cars (1, 2, 3, 4) 
Gender - Male/Female 
Married - Married/Single 
Loss in Thousands - Target variable """

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