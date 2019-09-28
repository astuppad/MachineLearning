# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:32:49 2019

@author: N827941
"""

import numpy as np
simple_list = [1,2,3]
np.array(simple_list)
list_of_lists = [[1,2,3], [4,5,6], [7,8,9]]
np.array(list_of_lists)
np.arange(0, 10)
np.arange(0, 21, 5)
np.zeros(100)
np.ones((4,5))
np.linspace(0,20, 10) #equally spaced values
np.eye(5) #identity matrix'
a = np.random.rand(3,2) #Uniform random distribution matrix 3X2
np.random.randn(2,3) #Normal random distribution mean = 0 standard deviation = 1 matrix 2X3
np.random.randint(1, 20, 10) #10 random integers between 1 and 20

sample_array = np.arange(30)
rand_array = np.random.randint(0, 100, 20)
sample_array.reshape(5,6) #converts in 5X6 matrix to hold all 30 integers from sample_array

rand_array.min()
rand_array.max()
rand_array.argmin() # Position/Index of min integer

sample_array.shape #gets the shape of the array
sample_array.reshape(1,30) 
sample_array.reshape(30, 1)

sample_array.dtype # gets the data type of an array

a.T # Gives the transpose of a matrix

sample_array = np.arange(10, 21)
sample_array[8]
sample_array[2: 5] # gets the values from index 2 to 4 5 will be ignored
sample_array[[2, 5]] # gets the values at the index postion 2 and 5 
    
#numpy array has capacity of broadcasting
sample_array = np.arange(10, 21)
subset_sample_array = sample_array[0:7] # creating view of sample_array
subset_sample_array[:] = 1001 # setting thes value to 1001 to every index is called broadcasting this subset of sample_array will be set to 1001 from 1 to 6th index

#to copy from one array to another
copy_of_array = sample_array.copy()
copy_of_array[:] = 10
copy_of_array

#two dimensional array
sample_matrix = np.array(([10,3,54,6], [4,7,32,15], [34,6,87,2], [21,17,2,9]))
sample_matrix[0][3]
sample_matrix[1,3]

sample_matrix[3:] #Fetch third row
sample_matrix[3]

sample_matrix[:, (1,3)] # all the rows but 2nd and third column
sample_matrix[:, (3,1)] # all the rows 3rd and 1st column ->sort columns


sample_array = np.arange(1, 31)
bool = sample_array < 10 # boolean array which array elements less than 10

sample_array[bool]
sample_array * sample_array
10/sample_array
sample_array + 1

# mathematical functions
np.exp(sample_array)
np.log(sample_array)
np.max(sample_array)
np.argmax(sample_array)
np.cos(sample_array)
np.sin(sample_array)
np.tan(sample_array)
np.square(sample_array)

# random matrix
array = np.random.randn(6, 6)
array
np.round(array, decimals = 2)
np.std(array) # standard devioation of an array
np.mean(array) # mean of an array

sports = np.array(['Cric', 'fball','cric','golf','cric', 'football' ])
np.unique(sports) # uniques values in an array

sample_array
simple_array = np.arange(0, 20)

simple_array

# Function to save python array into the file
np.save('simple_array', simple_array)

# Function to save multiple array into the zip file
np.savez('2_array.npz', a = sample_array, b = simple_array)

# load the data from the file (single array)
np.load('simple_array.npy')

#Load the data from the file (multiple array)
archive = np.load('2_array.npz')
archive['b']

# save and load the text files
np.savetxt('text_file.txt', simple_array, delimiter = ',')
np.loadtxt('text_file.txt')



