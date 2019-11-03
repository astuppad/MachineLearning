# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 08:22:13 2019

@author: N827941
"""

import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, mannwhitneyu, levene, shapiro, wilcoxon
from statsmodels.stats.power import ttest_power

energ = np.array([
# energy expenditure in mJ and stature (0=obese, 1=lean)
[9.21, 0],
[7.53, 1],
[7.48, 1],
[8.08, 1],
[8.09, 1],
[10.15, 1],
[8.40, 1],
[10.88, 1],
[6.13, 1],
[7.90, 1],
[11.51, 0],
[12.79, 0],
[7.05, 1],
[11.85, 0],
[9.97, 0],
[7.48, 1],
[8.79, 0],
[9.69, 0],
[9.68, 0],
[7.58, 1],
[9.19, 0],
[8.11, 1]])

# Seperating the data into 2 groups
group1 = energ[:, 1] == 0
group1 = energ[group1][:, 0] 
group2 = energ[:, 1] == 1
group2 = energ[group2][:, 0]

# two-sample t-test
# null hypothesis: the two groups have the same mean
# this test assumes the two groups have the same variance...
# (can be checked with tests for equal variance - Levene)
# independent groups: e.g., how boys and girls fare at an exam
# dependent groups: e.g., how the same class fare at 2 different exams
t_statistics, p_value = ttest_ind(group1, group2)
print(t_statistics, p_value)

# p_value < 0.05 => alternative hypothesis:
# they don't have the same mean at the 5% significance level
print("two-sample t-test p-value = ", p_value)

# two-sample wilcoxon test
# a.k.a Mann Whitney U - Used when samples are not normally distributed
u, p_value = mannwhitneyu(group1, group2)
print("two-sample wilconxon-test p-value = ", p_value)

# pre and post-surgery energy intake
intake = np.array([
[5260, 3910],
[5470, 4220],
[5640, 3885],
[6180, 5160],
[6390, 5645],
[6515, 4680],
[6805, 5265],
[7515, 5975],
[7515, 6790],
[8230, 6900],
[8770, 7335],
])

# Seperating data into 2 groups
pre = intake[:, 0]
post = intake[:, 1]

# paired t-test: doing two measurments on the same experimental unit
# e.g., before and after a treatment
t_statistics, p_value = ttest_1samp(post - pre, 0)
print("paired t-test p-value = ", p_value)

# alternative to paired t-test when data has an ordinary scale or when not
# normally distributed
z_statistics, p_value = wilcoxon(post - pre)
print("two-sample wilcoxon p-value = ", p_value)

# For checking equality of variance between groups
# Null Hypothesis: Variances are equal
levene(pre, post)

# For checking Normality distribution of each distribution
# Null Hypothesis: Distribution is Normal
shapiro(post)

# Calculating Power of Test
# Compute the difference in Means between 2 sample means and divide by pooled Standard Deviation 
# number of Observations/tuples
# Set the alpha value to 0.05 and alternative values 'two-sided' , 'larger' , 'smaller'
(np.mean(pre) - np.mean(post))/np.sqrt(((11-1)*np.var(pre)+(11-1)*np.var(post))/11+11-2)

print(ttest_power(0.87, nobs = 11, alpha = 0.05, alternative = 'two-sided'))

# Caculating power of test for the energ dataset
(np.mean(group1) - np.mean(group2)) / np.sqrt(((9 - 1) * np.var(group1) + (13 - 1) * np.var(group2)) / 9+13-1 )

print(ttest_power(effect_size = 0.57, nobs = 22, alpha = 0.10, alternative = 'two-sided'))