# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from scipy.stats import ttest_1samp

daily_intake = np.array([5260, 5470, 5640, 6180, 6390, 6515,
                         6805, 7515, 7515, 8230, 8770])

daily_intake.mean()

t_statistic, p_value = ttest_1samp(daily_intake, 7725)
print(t_statistic, p_value)

print("One-sample t-test p-value", p_value)