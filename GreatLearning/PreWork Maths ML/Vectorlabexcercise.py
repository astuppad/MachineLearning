# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:25:54 2019

@author: N827941
"""

import numpy as np

u = np.array([3, 4])
v = np.array([30, 40])

# Length / magnitude of the function

print(np.linalg.norm(u))
print(np.linalg.norm(v))

def direction(x):
   return x / np.linalg.norm(x)

w = direction(u)
z = direction(v)

w, z

# Direction vector always has a unit length 
print(np.linalg.norm(w))
print(np.linalg.norm(z))

import math

def geaometric_dot_product(x, y, theta):
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    return x_norm * y_norm * math.cos(math.radians(theta))

theta = 45 # reduce theta, dot product will increase
X = [3, 5]
y = [8, 2]

print(geaometric_dot_product(X, y, theta))

def dot_product(X, y):
    result = 0
    for i in range(len(X)):
        result = result + X[i] * y[i]
    return result        

print(dot_product(X, y)) 

X = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
A = np.vstack([X, np.ones(len(X))]).T
m, c = np.linalg.lstsq(A, y)[0]
print(m)

import matplotlib.pyplot as plt

plt.plot(X, m*X + c, 'b', label='line') # m is deltaX/deltaY
plt.legend()


