# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:55:45 2019

@author: N827941
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mean_pressure_compact_car    =  np.array([643, 655,702])
mean_pressure_midsize_car    =  np.array([469, 427, 525])
mean_pressure_fullsize_car   =  np.array([484, 456, 402])

print('Count, Mean and standard deviation of mean pressue exerted by compact car: %3d, %3.2f and %3.2f' % (len(mean_pressure_compact_car ), mean_pressure_compact_car .mean(),np.std(mean_pressure_compact_car ,ddof =1)))
print('Count, Mean and standard deviation of mean pressue exerted by midsize car: %3d, %3.2f and %3.2f' % (len(mean_pressure_midsize_car), mean_pressure_midsize_car.mean(),np.std(mean_pressure_midsize_car,ddof =1)))
print('Count, Mean and standard deviation of mean pressue exerted by full size car: %3d, %3.2f and %3.2f' % (len(mean_pressure_fullsize_car), mean_pressure_fullsize_car.mean(),np.std(mean_pressure_fullsize_car,ddof =1)))

mean_pressure_df = pd.DataFrame()

df1 = pd.DataFrame({'Car_Type': 'C', 'Mean_Pressure': mean_pressure_compact_car})
df2 = pd.DataFrame({'Car_Type': 'M', 'Mean_Pressure': mean_pressure_midsize_car})
df3 = pd.DataFrame({'Car_Type': 'F', 'Mean_Pressure': mean_pressure_fullsize_car})

mean_pressure_df = mean_pressure_df.append(df1)
mean_pressure_df = mean_pressure_df.append(df2)
mean_pressure_df = mean_pressure_df.append(df3)

sns.boxplot(x = 'Car_Type', y = 'Mean_Pressure', data = mean_pressure_df)
plt.title('Mean pressure exerted by car types')

import statsmodels.api as sm
from statsmodels.formula.api import ols

mod = ols('Mean_Pressure ~ Car_Type', data = mean_pressure_df).fit()
aov_table = sm.stats.anova_lm(mod, typ = 2)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
print(pairwise_tukeyhsd(mean_pressure_df['Mean_Pressure'], mean_pressure_df['Car_Type']))

# Two way anova
table1  = [['Day','Store-A','Store-B','Store-C','Store-D','Store-E'], [1,79, 81, 74, 77, 66],\
           [2, 78, 86, 89, 97, 86], [3, 81, 87, 84, 94, 82], [4, 80, 83, 81, 88, 83], [5, 70, 74, 77, 89, 68]]

headers = table1.pop(0) 
df1 = pd.DataFrame(table1, columns =  headers)

d0_val  = df1['Day'].values
d1_val  = df1['Store-A'].values
d2_val  = df1['Store-B'].values
d3_val  = df1['Store-C'].values
d4_val  = df1['Store-D'].values
d5_val  = df1['Store-E'].values

df1     = pd.DataFrame({'Day': d0_val, 'Store':'A', 'QoS': d1_val})
df2     = pd.DataFrame({'Day': d0_val, 'Store':'B', 'QoS': d2_val})
df3     = pd.DataFrame({'Day': d0_val, 'Store':'C', 'QoS': d3_val})
df4     = pd.DataFrame({'Day': d0_val, 'Store':'D', 'QoS': d4_val})
df5     = pd.DataFrame({'Day': d0_val, 'Store':'E', 'QoS': d5_val})

QoS_df  = pd.DataFrame()

QoS_df  = QoS_df.append(df1) 
QoS_df  = QoS_df.append(df2) 
QoS_df  = QoS_df.append(df3) 
QoS_df  = QoS_df.append(df4) 
QoS_df  = QoS_df.append(df5) 

pd.DataFrame(QoS_df)

from statsmodels.stats.anova import anova_lm

formula = 'QoS ~ C(Day) + C(Store)'
model = ols(formula, QoS_df).fit()
aov_table = anova_lm(model, typ = 2)

# Chi - square test
import scipy.stats as stats
import scipy 

observed_values = scipy.array([190, 185, 90, 35])
n = observed_values.sum()

expected_values = scipy.array([n*0.30, n*0.45, n*0.20, n*0.05])
chi_square_test, p_value = stats.chisquare(observed_values, f_exp = expected_values)

print('At 5 %s level of significance, the p-value is %1.7f' %('%', p_value))
# There we reject the null hypothesis


expected_values2 = scipy.array([n*0.28, n*0.42, n*0.25, n*0.05])  
expected_array = np.array([expected_values, expected_values2])  
chi_sq_stat, p_value, deg_freedom, exp_freq = stats.chi2_contingency(expected_array)
(chi_sq_stat, p_value, deg_freedom, exp_freq)
# we accept the null hypothesis


quality_array = np.array([[138, 83, 64],[64, 67, 84]])
chi_sq_Stat, p_value, deg_freedom, exp_freq = stats.chi2_contingency(quality_array)

print('Chi-square statistic %3.5f P value %1.6f Degrees of freedom %d' %(chi_sq_Stat, p_value,deg_freedom))
# we reject the null hypothesis in this case.