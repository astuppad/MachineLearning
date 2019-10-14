# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:43:34 2019

@author: N827941
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv('Inc_Exp_Data.csv')

dataset['Mthly_HH_Expense'].mean()

dataset['Mthly_HH_Expense'].median()

mthly_exp_tmp = pd.crosstab(index = dataset['Mthly_HH_Expense'], columns = 'count', aggfunc = 'count', values = dataset['Mthly_HH_Expense'])
mthly_exp_tmp.reset_index(inplace = True)
mthly_exp_tmp[mthly_exp_tmp['count'] == dataset['Mthly_HH_Expense'].value_counts().max()] 

dataset['Highest_Qualified_Member'].value_counts().plot(kind = 'bar')

dataset.plot(x = 'Mthly_HH_Income', y = 'Mthly_HH_Expense')
IQR = dataset['Mthly_HH_Income'].quantile(0.75) - dataset['Mthly_HH_Expense'].quantile(0.25)
IQR

pd.DataFrame(dataset.iloc[:, 0:5].std().to_frame()).T

pd.DataFrame(dataset.iloc[:, 0:4].std().to_frame()).T

pd.DataFrame(dataset['Highest_Qualified_Member'].value_counts().to_frame()).T

dataset['No_of_Earning_Members'].value_counts().plot(kind='bar')

