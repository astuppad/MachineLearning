# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:43:22 2019

@author: N827941
"""

import numpy as np

array = np.zeros(20)
array[3] = 5

array_master = np.ones(20)
array_master[1] = 1
array_copy = array_master.copy()

array_broadcast = np.ones(30)
array_broadcast[:] = 100

array1 = np.arange(21, 32)
array2 = np.arange(11, 22)
difference = array1 - array2

a1 = np.arange(2, 11, 2)
a2 = np.arange(22, 31, 2 )

np.stack((a1, a2))
np.stack((a1, a2), axis = 1)
np.column_stack((a1, a2)) # both works the same way

matrix = np.arange(0, 30).reshape(5, 6)
value = matrix[1][2]

identity_matrix = np.eye(10)
identity_matrix[identity_matrix == 0] = 21

random_array = np.random.rand(20)
random_array > 0.2

np.random.randn(5,2)
np.linspace(0, 100, 30) # linearly spaced points

# Numpy Indexing and Selection
simple_matrix = np.arange(1,101).reshape(10, 10)
simple_matrix[8:, 0:3]
simple_matrix[5, 4]
simple_matrix[:, 2]
simple_matrix[3, :]
simple_matrix[(2,4), :]

# Calculation through numpy functions
np.sum(simple_matrix)
np.std(simple_matrix)
np.var(simple_matrix)
np.mean(simple_matrix)
np.max(simple_matrix)
np.min(simple_matrix)
