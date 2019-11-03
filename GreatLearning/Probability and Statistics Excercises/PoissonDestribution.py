# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:43:26 2019

@author: N827941
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

rate = 3
n=np.arange(0, 20)
poisson = stats.poisson.pmf(n, rate)

poisson[4]

1 - (poisson[0]+poisson[1]+poisson[2]+poisson[3])

plt.plot(n, poisson, '-o')
plt.title('Poisson: $\lambda$ = %i' % rate )
plt.xlabel('Number of customers arriving in a minute')
plt.ylabel('Probability of Number of customers arriving in a minute')