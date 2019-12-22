# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 18:56:58 2019

@author: N827941
"""
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 200)
# Reading the dataset
loan_df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
loan_df.shape
loan_df.head()

# Analyzing the spread of the data
sns.pairplot(loan_df, diag_kind = 'kde')

# Analyzing a correlation and spread of the data between all variables
loan_df.corr()
loan_df.info()
loan_df.describe().transpose()

# plotting the correlation plot
corr_matrix = pd.DataFrame(loan_df.corr())
fig , ax = plt.subplots(figsize = (corr_matrix.count(axis = 1)[0], corr_matrix.count(axis = 1)[0]))
ax.matshow(corr_matrix)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# distribution of target variable
sns.distplot(loan_df['CreditCard'], kde = True)

# defining X and y variable for model building
X = loan_df.drop(['Personal Loan', 'ID'], axis = 1)
y = loan_df['Personal Loan']

# Splitting the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)

# apply the standard scaler for higly varying magnitude attrubutes
# without feature scaling we achieved the 90% accuracy
# with feture scaling we acheived 94% accuracy
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
cols = X_train.columns
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Calculate the usage of credit card issued by the universal bank
n_true, n_false = loan_df['Personal Loan'].value_counts()
print('Number of true cases {0} ({1: 0.2f})'.format(n_false, (n_false / (n_true + n_false)))) # 10% false
print('Number of false cases {0} ({1: 0.2f})'.format(n_true, (n_true / (n_true + n_false)))) # 90% true

# train logistice regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'liblinear')
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

# find the accuracy of model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = ['1', '0'],
                         columns = ['Predicted 1', 'Predicted 0'])
plt.figure(figsize = (7, 5))
sns.heatmap(df_cm, annot = True)

# train the model using KNN classifier
from sklearn.neighbors import KNeighborsClassifier
scores = []
for i in range(1, 20):
    knnclassifier = KNeighborsClassifier(n_neighbors = i, weights = 'distance', metric = 'euclidean')
    knnclassifier.fit(X_train, y_train)
    y_pred = knnclassifier.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    
plt.plot(range(1, 20), scores) # so the conclusion is neghbors selection should be between 0 to 7 to perform good accuracy of the model

np.max(scores) # to get the highest accuracy of the model

# training the model using naive bayes
from sklearn.naive_bayes import BernoulliNB
BernoNB = BernoulliNB()
BernoNB.fit(X_train, y_train)
y_pred = BernoNB.predict(X_test)

# find the accuracy of Naive-Bayes 
accuracy_score(y_test, y_pred)

# The dataset is being tested with 3 models i,e. LogisticRegression, KNeighborsClassifier and NaiveBayes
# Logistic regression predicted with accuracy of 90.4%
# KNeighborsClassifier predicted with accuracy of 90.7
# NavieBayes predicted with accuracy of 89.2% 

# lets start eliminating the features and check for the performance
import statsmodels.api as sm
regressor = sm.OLS(y_train, X_train).fit()

# We will get the features that are greater than significance level of 0.05 to eliminate the features which are effecting very less to build an model
rejected_features = []
for index in range(len(regressor.pvalues)):
    if regressor.pvalues[index] > 0.05:
        rejected_features.append(index)

# we got columns 0, 1, 3 pvalues > 0.05 
# remove the fearures selected from the backword eliminatation method
X_train = pd.DataFrame(X_train).drop(rejected_features, axis = 1)
X_test = pd.DataFrame(X_test).drop(rejected_features, axis = 1)

# Add the columns headers back to training set and test set
X_train.columns = np.delete(cols, [0, 1, 3])
X_test.columns = np.delete(cols, [0, 1, 3])

regressor = LogisticRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
accuracy_score(y_test, y_pred)

# we still achieved 95% accuaracy by eliminating --- attributes
# try all the model from the start and find the best fit model
# lets choose KNN classifier with 96.5% accuracy after feature selection and scaling the attributes