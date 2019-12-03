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
from scipy.stats import skew

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

# g. Checking the presence of outliers in ‘bmi’, ‘age’ and  ‘charges columns
# count of the outliers of bmi.
q1, q3 = np.percentile(insurance.bmi, [25, 75])
iqr_bmi = q3 - q1
lower_bound_bmi = q1 - 1.5 * iqr_bmi
upper_bound_bmi = q1 + 1.5 * iqr_bmi
insurance[(insurance.bmi < lower_bound_bmi) | (insurance.bmi > upper_bound_bmi)].bmi.count() # 130

# count of the outliers of age.
q1, q3 = np.percentile(insurance.age, [25, 75])
iqr_age = q3 - q1
lower_bound_age = q1 - 1.5 * iqr_age
upper_bound_age = q1 + 1.5 * iqr_age
insurance[(insurance.age < lower_bound_age) | (insurance.age > upper_bound_age)].age.count() #22

# count of the outliers of charges.
q1, q3 = np.percentile(insurance.charges, [25, 75])
iqr_charges = q3 - q1
lower_bound_charges = q1 - 1.5 * iqr_charges
upper_bound_charges = q1 + 1.5 * iqr_charges
insurance[(insurance.charges < lower_bound_charges) | (insurance.charges > upper_bound_charges)].charges.count() #230 

# Visualising the outliers of bmi, age and charges
sns.boxplot(insurance['bmi'])
sns.boxplot(insurance['age'])
sns.boxplot(insurance['charges'])

# Distribution of categorical columns (include  children) 
sns.swarmplot(x = 'sex', y = 'charges', data = insurance)
sns.swarmplot(x = 'children', y = 'charges', data = insurance)
sns.swarmplot(x = 'smoker', y = 'charges', data = insurance)
sns.swarmplot(x = 'region', y = 'charges', hue='sex', data = insurance)

# a. Do charges of people who smoke differ significantly  from the people who don't?  
# To answer this we set hypothesis
# H0: Non smokers charges = Smokers charges
# H1: Non smokers charges != Smokers charges  
# we divide charges into 2 groups i,e. Smokers and Nonsmokers
smokers = insurance[insurance.smoker == 'yes'].charges
nonsmokers = insurance[insurance.smoker == 'no'].charges

# First we check if the data is normally distributed
sns.distplot(smokers)
sns.distplot(nonsmokers)

smokers.mean()
nonsmokers.mean()

smokers.var() == nonsmokers.var()
from scipy.stats import mannwhitneyu, ttest_ind
t_statistics, p_value = ttest_ind(smokers,nonsmokers) # p_value > 0.05 
# Therefore the data is not normally distributed so we use mannwhitneyu test
u, p_value = mannwhitneyu(smokers, nonsmokers) # p_value > 0.05
# Therefore, we accept the the null hypothysis since the smoker and non smokers charges are not equal

# b. Does bmi of males differ significantly from that of  females?
#H0: female bmi = male bmi
#H1: female bmi != male bmi
# we divide bmi into 2 groups, female and male
male_bmi = insurance[insurance.sex == 'male'].bmi
female_bmi = insurance[insurance.sex == 'female'].bmi

male_bmi.mean() == female_bmi.mean()
male_bmi.var() == female_bmi.var()
t_statistics, p_value = ttest_ind(male_bmi, female_bmi)
# the data is not normally distributed so we use manwhitneyu/wicoxen test
u, p_value = mannwhitneyu(male_bmi, female_bmi)
# male and female slightly greater than 0.05 significance level, So we reject the null hypthesis at 0.05 significance level

# c. Is the proportion of smokers significantly different  in different genders? 
from statsmodels.stats.proportion import proportions_ztest
female_smokers = insurance[insurance.sex == 'female'].smoker.value_counts()[1]
male_smokers = insurance[insurance.sex == 'male'].smoker.value_counts()[1]
n_males, n_females = insurance.sex.value_counts()

stat, p_value = proportions_ztest([female_smokers, male_smokers], [n_females, n_males])

if(p_value < 0.05):
    print(f'With a p-value of {round(p_value, 4)} is significantly different Therefore, we reject the null hypothesis')
else:
    print(f'With a p-value of {round(p_value, 4)} is not significantly different Therefore, we accept the null hypothesis')


# d. Is the distribution of bmi across women with no  children, one child and two children, the same ? 
childrenbmi = insurance[insurance['sex'] == 'female'].loc[:, ['children', 'bmi']]
childrenbmi = childrenbmi[childrenbmi.children < 3]
import statsmodels.api as sm
from statsmodels.formula.api import ols

sns.boxplot(x = 'children', y ='bmi', data = childrenbmi)

mod = ols('bmi ~ children', data = childrenbmi).fit()
aov_table = sm.stats.anova_lm(mod, typ = 2)
p_value = round(aov_table['PR(>F)'][0], 4)
if(p_value < 0.05):
    print(f' With p-value {p_value} is less than the choosen significance value 0.05, Therefore we reject the null hypothesis')
else:
    print(f' With p-value {p_value} is not less than the choosen significance value 0.05, Therefore we accept the null hypothesis')

