# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:44:26 2019

@author: N827941
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import seaborn as sns

insurance = pd.read_csv('insurance.csv')

sns.distplot(insurance['charges'])
sns.barplot(insurance['sex'], insurance['charges'])
sns.countplot(insurance['sex'])
sns.jointplot(insurance['age'], insurance['charges'])

# chanrges based on sex
plt.hist(insurance['charges'])
plt.bar(insurance['sex'], insurance['charges'])

#charges based on smokers and non smokers
sns.barplot(insurance['smoker'], insurance['charges'])

# insurance charges based on age sex and smoker
sns.pairplot(insurance, x_vars = ['age', 'sex', 'smoker'], y_vars = 'charges')

# distrubution of charges based on bmi
sns.swarmplot(x = 'bmi', y = 'charges', data = insurance)

# distribution based on childrens
sns.scatterplot(x = 'children', y = 'charges', data = insurance)

# count of smoker based on regions
sns.countplot(x = 'region', hue = 'smoker', data = insurance)

# count of male smokers and female smokers
sns.countplot(x = 'sex', hue = 'smoker', data = insurance[insurance.smoker == 'yes'])

# a
# shape of the data
sns.distplot(insurance['charges'])

# b
# data type for each attribut
insurance.info()

# c
# Checking the presence of missing values
insurance.isnull().values.any()

# d
# 5 number summary for BMI
sns.boxplot(x = 'sex', y = 'bmi', data = insurance)

# e 
# Distribution of ‘bmi’, ‘age’ and ‘charges’ columns.
sns.distplot(a = insurance['bmi'])
sns.distplot(a = insurance.age)
sns.distplot(a = insurance.charges)

sns.boxplot(x = 'sex', y = 'charges', hue = 'smoker',data = insurance[['sex', 'charges', 'smoker']])

# f. Measure of skewness of ‘bmi’, ‘age’ and ‘charges’  columns
# To measure the skewness of the data we have the skewness   formula as 
# mean - median / standard deviation as per the pearson median skewness
# for the bmi we calculate standard deviation, mean and median as
insurance.bmi.mean()
insurance.bmi.median()
insurance.bmi.std()
# so when we apply the formula to calculate the skewness
skewness_bmi = (3*(insurance.bmi.mean() - insurance.bmi.median())) / insurance.bmi.std()

# plot the diagram for the bmi
sns.distplot(insurance.bmi, hist = False, rug = True)

# measure of skewness of data from scipy.stats.skew
from scipy.stats import skew
skewness_bmi = skew(insurance.bmi,  axis = 0, bias = False)

# plot the diagram for the age
sns.distplot(insurance.age, hist = False)

# measure the skewness of age fro msipy.stats.skew
skewness_age = skew(insurance.age,  axis = 0, bias = False)

# plot the diagram for the charges
sns.distplot(insurance.charges, hist = False)

# measure the skewness of age fro msipy.stats.skew
skewness_charges = insurance.charges.skew(axis = 0, skipna = True)


# know the skewness is left or right
if (skewness_charges > 0):
    print('Charges is left skewed')
elif(skewness_charges < 0):
    print('Charges is right skewed')
else:
    print('Charges is normally distributed')
    




