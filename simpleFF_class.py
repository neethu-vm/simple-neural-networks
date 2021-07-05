#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:17:41 2021

@author: christy
"""


import numpy as np

def sigmoid(x):
  # Our activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def reLU(x):
    if x<=0:
        return 0
    else:
        return x

class Neuron:
  def __init__(self, w, b):
    self.weights = w
    self.bias = b

  def feedforward(self, inputs):
    # Weight inputs, add bias, then use the activation function
    total = np.dot(self.weights, inputs) + self.bias
    return sigmoid(total)
 
    
class NeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  Each neuron has the same weights and bias:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = np.array([0, 1])
    bias = 0

    # The Neuron class here is from the previous section
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # The inputs for o1 are the outputs from h1 and h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x))


#w = np.array([2, 1]) # w1 = 0, w2 = 1
#b = 4                   # b = 0
#NN = Neuron(w, b)#

#NN1 = Neuron(np.array([2,2]),b)

#x = np.array([2, 3])       # x1 = 2, x2 = 3
#print(NN.feedforward(x))