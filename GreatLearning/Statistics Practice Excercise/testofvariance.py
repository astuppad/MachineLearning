# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 10:13:08 2019

@author: N827941
"""

import numpy as np
import pandas as pd
import scipy.stats as f
import matplotlib.pyplot as plt

df = pd.read_csv('insurance.csv')

df.head()

# Test of proportions
female_smokers = df[df['sex'] == 'female'].smoker.value_counts()[1] # number of female smokers
male_smokers = df[df['sex'] == 'male'].smoker.value_counts()[1] # number of male smokers
n_females = df[df.sex == 'female'].sex.count() # total number of females
n_males = df[df.sex == 'male'].sex.count() # total number of males

print([female_smokers, male_smokers], [n_females, n_males])
print(f'Proportion of smokers in females, males = {round(115/662, 2)}, {round(159/676)} respectively')

from statsmodels.stats.proportion import proportions_ztest

stat, pval = proportions_ztest([female_smokers, male_smokers], [n_females, n_males])

if pval < 0.5:
    print(f'With a p-value of {round(pval, 4)} the difference is significant, aka |We reject the null hypothesis|')
else:
    print(f'With a p-value of {round(pval, 4)} the difference is not significant, aka |We accept the null hypothesis|')
    

nineteen = df[df.age == 19]
nineteen.sex.value_counts()

sample_male = nineteen[nineteen.sex == 'male'].bmi.iloc[:-2] # excluding the last 2 elements to match the size 2 samples
sample_female = nineteen[nineteen.sex == 'female'].bmi

v1, v2 = np.var(sample_female), np.var(sample_male)

n = 33 #number of samples
dof = n - 1 # degrees of freedom
alpha = 0.05 # significance of value
chi_critical = 	46.19 # critival chi squared value, From the table

chi = (dof * v1)/ v2
if(chi < chi_critical):
    print('Since the test is statistically less than the critical value, we fail to reject the null')
else:
    print('Since the test is statistically more than the critical value, we reject the null')
