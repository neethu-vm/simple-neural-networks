#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 08:38:46 2021

@author: christy
"""

import numpy as np

def feedforward(inputs,weights,bias):
    y = np.dot(inputs,weights)+bias
    return y


def sigmoid(y):
    y_op = 1/(1+np.exp(-y))
    return y_op

def relu(y):
    if y <=0:
        return 0
    else:
        return y

def activation(y):
    return y

inputs = np.array([1,4,5])

h1_weights = np.array([0.1,0.3,0.5])
h2_weights = np.array([0.2,0.4,0.6])
o1_weights = np.array([0.7,0.9])
o2_weights = np.array([0.8,0.1])

b1 = 0.5
b2 = 0.5

h1 = feedforward(inputs,h1_weights,b1)
h2 = feedforward(inputs,h2_weights,b1)

h1 = relu(h1)
h2 = relu(h2)

print("h1:",h1)
print("h2:",h2)



h_outputs = np.array([h1,h2])

o1 = feedforward(h_outputs,o1_weights,b2)
o2 = feedforward(h_outputs,o2_weights,b2)

o1 = relu(o1)
o2 = relu(o2)

print("o1:",o1)
print("o2:",o2)
    