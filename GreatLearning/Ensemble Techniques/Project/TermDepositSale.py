# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:23:59 2020

@author: Niket
"""
# import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
# Import the bank dataset 
bank = pd.read_csv( 'bank-full.csv', sep = ';')

bank.head(10)

# Analysing univariate data
bank.describe().transpose()

# Analyszing multivariate data
correlation_matrix = bank.corr()

# plotting the correlation matrix to analyze the relation between the attributes
colormap = plt.cm.viridis   
plt.figure(figsize = (15,15))
sns.heatmap(bank.corr(), linewidths= 0.1, vmax = 1.0, 
            square = True, cmap = colormap, linecolor = 'white', annot = True)
# by the correlation plot we got campaign, day and campaign have weak relationships 

# anayzing the data by pairplot
sns.pairplot(bank)
#looks there is very week relationship with attribute 'day' and it may week for predictions so we remove this attribute later.


bank.info()
# since,lbank_encoded = pd.get_dummies(bank)ot of atrtributes are objecct, and some of the attributes are categorical we make them categorical

# we plot the boxplot for the balance and duration attributes
sns.boxplot(bank['balance'])
sns.boxplot(bank['duration'])
sns.boxplot(bank['pdays'])

#since, the balance and duration has lot of outliers 
# we remove the outliers from the dataset
from scipy import stats

z = np.abs(stats.zscore(bank[['age', 'balance', 'day', 'duration', 'campaign', 'previous','pdays']]))
print(z)

Q1 = bank.quantile(0.25)
Q3 = bank.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

#print((bank < (Q1 - 1.5 * IQR)) | (bank > (Q3 + 1.5 * IQR)))

bank_refined = bank[(z < 3).all(axis = 1)]

# we see how many rows we have deleted from the dataset whech were outliers
print(bank.shape)
print(bank_refined.shape)


# but we have pdays attribute which is having missing values 
bank['pdays'].head()

# we get the categorical variables 
bank_refined['job'].unique()
bank_refined['job'] = bank_refined['job'].replace({'unemployed' : 1, 'services' : 2, 'management': 3, 'blue-collar' : 4,
       'self-employed': 5, 'technician': 6, 'entrepreneur': 7, 'admin.': 8, 'student': 9,
       'housemaid': 10, 'retired': 11, 'unknown': 0})
bank_refined['month'] = bank_refined['month'].replace({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
                        'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})

bank_categorical = pd.get_dummies(bank_refined, columns = ['marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome'], drop_first = True)

bank_categorical.info()

# define X and y as independent and dependent variables
X = bank_categorical.drop('y', axis = 1)
y = bank_categorical['y']

# check the distribution of y variable.
plt.figure(figsize = (7, 7))
sns.distplot( a=y.value_counts(), hist = True, kde = True, color = 'green')
y.value_counts()


# split the data for training and testing by 70:30 split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# fit the first ensemble model DicisionTree
maxScore = 0
maxFeature = 0
from sklearn.tree import DecisionTreeClassifier
bank_classifier = DecisionTreeClassifier(max_features = 16, min_samples_split = 3, min_impurity_split = 0.12, criterion= 'gini', max_depth = 5)
bank_classifier.fit(X_train, y_train)
print(bank_classifier.score(X_train,y_train))
print(bank_classifier.score(X_test, y_test))


# base classifier perfomrd good in training data set with 92% accuracy whereas in test dataset it achived 91%
# this result achieved by tuning depth parmeter

# so we try with scaling the data 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.fit_transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)

# trying the Decision tree model with scaled dataset
bank_classifier.fit(X_train_scaled, y_train)
print(bank_classifier.score(X_train_scaled, y_train))
print(bank_classifier.score(X_test_scaled, y_test))
# this still achieve the same result 

# lets see the decision tree model 
import graphviz
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import pydotplus

dot_data = tree.export_graphviz( bank_classifier, out_file = None,
                                feature_names = list(X_train),
                                class_names = ["No", "Yes"],
                                rounded = True,
                                special_characters = True,
                                )
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# we try with different ensemble techniques to tune and correct the accuracy
from sklearn.ensemble import BaggingClassifier
bag_classifier = BaggingClassifier(base_estimator = bank_classifier, n_estimators = 50, max_features = 22, random_state = 10)
bag_classifier.fit(X_train, y_train)
print(bag_classifier.score(X_train, y_train))
print(bag_classifier.score(X_test, y_test))

# ada boosting 
from sklearn.ensemble import AdaBoostClassifier
ada_classifier = AdaBoostClassifier(base_estimator= bank_classifier, n_estimators= 100, random_state = 10)
ada_classifier.fit(X_train, y_train)
print(ada_classifier.score(X_train, y_train))
print(ada_classifier.score(X_test, y_test))

# gradient ada boostring
from sklearn.ensemble import GradientBoostingClassifier
grad_classifier = GradientBoostingClassifier(loss = 'deviance', n_estimators = 10)
grad_classifier.fit(X_train_scaled, y_train)
print(grad_classifier.score(X_train, y_train))
print(grad_classifier.score(X_test, y_test))

# random forest
from sklearn.ensemble import RandomForestClassifier
random_classifier = RandomForestClassifier(n_estimators='warn', criterion = 'gini', max_depth = 15, 
                                           max_features=16)
random_classifier.fit(X_train, y_train)
print(random_classifier.score(X_train_scaled, y_train))
print(random_classifier.score(X_test_scaled, y_test))


import sklearn.svm as svm
svc = svm.SVC(gamma= 0.01, C = 100)
svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))
print(svc.score(X_test, y_test))


