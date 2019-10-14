# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:14:31 2019

@author: N827941
"""

def greetings(name):
    """ This fucntion greets a person
    whos name is passed as a paramter to a function call
    as parameter"""
    print("Hello," + name + ". How are you doing")

greetings("Niket")
greetings.__doc__

def absolute(num):
    if num >= 0:
        return(num)
    if num < 0:
        return(abs(num))
        
absolute(-4)

# scope of a variable is the portion of the program within which the variable is recognized.
def my_func():
    x = 10
    print("Value inside function", x)

x = 20
my_func()
print("Value outside function", x)

# The scope of a variable defined inside a function is the function boundary
# The moment a fucntion is exited, the variable removed from the memory. Liftime of the
# variable in the function is only as long as the function is being executed.

# To access variable defined outside the function within  a function, we use global declaration
x = 20

def my_func():
    global x
    x = 10
    print("Value inside function", x)

my_func()
print("Value outside function", x)

def add_numbers(x, y):
    sum = x + y
    return sum

num1 = 5
num2 = 6
print("The sum is ", add_numbers(num1, num2))