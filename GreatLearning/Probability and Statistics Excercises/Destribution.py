# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:33:14 2019

@author: N827941
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

n=7
p=.6
k=np.arange(0, 8)

binomial = stats.binom.pmf(k, n, p)

plt.plot(k, binomial, '-o')
plt.title('Binomial: n=%i, p=%.2f' % (n,p), fontsize = 15)
plt.xlabel('Number of success')
plt.ylabel('Probability of success')