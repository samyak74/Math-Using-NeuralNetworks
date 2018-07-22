# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:10:53 2018

@author: Sam

A simple neural network to recognize patterns.
Sigmoid activation function used.
3 inputs, 1 output.
The pattern is output being same as the first column of input.
"""
import math
from numpy import exp,array,random,dot

class neural_networks:
    def __init__(self):
        random.seed(1)
        
        self.weights= 2* random.random((3,1))-1
        
    def __sigmoid(self,x):
        return 1/(1+exp(-x))
    
    def train(self,inputs,outputs,num):
        for iteration in range(num):
            output=self.think(inputs)
            error= outputs-output
            adjustment= dot(inputs.T,error*output*(1-output))
            self.weights+=adjustment
            
    def think(self,inputs):
        result= self.__sigmoid(dot(inputs,self.weights))
        return result
    
    
network= neural_networks()

# the training set
inputs= array([[1,1,1],[1,0,1],[0,1,1]])
outputs=array([[1,1,0]]).T

#training
network.train(inputs,outputs,10000)

print((network.think(array([1,0,0]))))