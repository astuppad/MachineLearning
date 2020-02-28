# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:37:09 2020

@author: Niket
"""

"""
Income prediction on census data

Objective:
To predict whether income exceeds 50K/yr based on census data

Dataset: Adult Data Set
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

Variable description:
age: continuous
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
class: >50K, <=50K
"""

# Pandas and Numpy labraries.
import pandas as pd
import numpy as np

# For preprocessing the data
from sklearn.preprocessing import Imputer
from sklearn import preprocessing

# To split the data into train and test data
from sklearn.model_selection import train_test_split

# To model the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score

adult_df = pd.read_csv('adult.data', header = None, delimiter = ' *, *', engine = 'python')

"""
Load the dataset. Observe that this file has .data extention
For importing the census data, we are using pandas read_csv() method. This method is a very simple and fast method for importing data.
We are passing four parameters. The ‘adult.data’ parameter is the file name. The header parameter is for giving details to pandas that whether the first row of data consists of headers or not. In our dataset, there is no header. So, we are passing None.
The delimiter parameter is for giving the information the delimiter that is separating the data. Here, we are using ‘ , ’ delimiter. This delimiter is to show delete the spaces before and after the data values. This is very helpful when there is inconsistency in spaces used with data values.
"""

# Adding header to the dataset
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

# Number of records in the dataframe
len(adult_df)

# Handling missing data
# Test whether there is any null value in out dataset or not. We can d o this using isnull() method.
adult_df.isnull().sum()

"""
The above output shows that there is no “null” value in our dataset.
Let’s try to test whether any categorical attribute contains a “?” in it or not. At times there exists “?” or ” ” in place of missing values. Using the below code snippet we are going to test whether adult_df data frame consists of categorical variables with values as “?”.
"""

for value in ['workclass','education','marital_status','occupation','relationship','race','sex','native_country','income']:
    print(value, ":", sum(adult_df[value] == '?'))
    

# The output of the above code snippet shows that there are 1836 missing values in workclass attribute. 1843 missing values in occupation attribute and 583 values in native_country attribute.


"""Data preprocessing
For preprocessing, we are going to make a duplicate copy of our original dataframe.We are duplicating adult_df to adult_df_rev dataframe. Observe that we have used deep copy while copying. Why?"""

## Deep copy of adult_df
adult_df_rev = adult_df.copy(deep = True)

#Before doing missing values handling task, we need some summary statistics of our dataframe. For this, we can use describe() method. It can be used to generate various summary statistics, excluding NaN values.
adult_df_rev.describe()

# We are passing an “include” parameter with value as “all”, this is used to specify that. we want summary statistics of all the attributes.
adult_df_rev.describe(include='all')

"""
Data imputation
Some of the categorical values have missing values i.e, “?”. We replace the “?” with the above describe methods top row’s value. For example, we replace the “?” values of workplace attribute with “Private” value.
"""

for value in ['workclass', 'education', 'marital_status', 'relationship', 'race', 'sex', 'native_country', 'income']:
    replaceValue = adult_df_rev.describe(include='all')[value][2]
    adult_df_rev[value][adult_df_rev[value] == '?'] = replaceValue


"""For Naive Bayes, we need to convert all the data values in one format.
We are going to encode all the labels with the value between 0 and n_classes-1. In the present case, it will be 0 and 1.
For implementing this, we are going to use LabelEncoder of scikit learn library."""

# Hot Encoding
le = preprocessing.LabelEncoder()
workclass_cat = le.fit_transform(adult_df.workclass)
education_cat = le.fit_transform(adult_df.education)
marital_cat = le.fit_transform(adult_df.marital_status)
occupation_cat = le.fit_transform(adult_df.occupation)
relationship_cat = le.fit_transform(adult_df.relationship)
race_cat = le.fit_transform(adult_df.race)
sex_cat = le.fit_transform(adult_df.sex)
native_country_cat = le.fit_transform(adult_df.native_country)

# initialize the encoded categorical columns
adult_df_rev['workclass_cat'] = workclass_cat
adult_df_rev['education_cat'] = education_cat
adult_df_rev['marital_cat'] = marital_cat
adult_df_rev['occupation_cat'] = occupation_cat
adult_df_rev['relationship_cat'] = relationship_cat
adult_df_rev['race_cat'] = race_cat
adult_df_rev['sex_cat'] = sex_cat
adult_df_rev['native_country_cat'] = native_country_cat

adult_df_rev.head()
dummy_fields = ['workclass', 'education', 'marital_status', 'relationship', 'race', 'sex', 'native_country']
adult_df_rev = adult_df_rev.drop(dummy_fields, axis = 1)

adult_df_rev = adult_df_rev.reindex_axis(['age', 'workclass_cat', 'fnlwgt', 'education_cat',
                                    'education_num', 'marital_cat', 'occupation_cat',
                                    'relationship_cat', 'race_cat', 'sex_cat', 'capital_gain',
                                    'capital_loss', 'hours_per_week', 'native_country_cat', 
                                    'income'], axis = 1)
adult_df_rev.head()
#Now we have created multiple categorical columns like “marital_cat”, “race_cat” etc. !

# Data Slicing
# Arrange data into independent variables and dependent variables
X = adult_df_rev.values[: , :14] ## Features
y = adult_df_rev.values[:, 14] ## Target

# Split the data into train and test
# Train data size: 70% of original data
# Test data size: 30% of original data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

# Implement Gaussian Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

# Now GaussianNB classifier is built. The classifier is trained using training data. We can use fit() method for training it. After building a classifier, our model is ready to make predictions. We can use predict() method with test set features as its parameters.
y_pred = clf.predict(X_test)

# Accuracy of our Gaussian Naive Bayes model
accuracy_score(y_test, y_pred, normalize = True)


















