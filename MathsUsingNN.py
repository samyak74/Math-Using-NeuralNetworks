# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:40:33 2018

@author: Sam

*Required Numpy Library only

This is a simple single layer neural network which automatically learns the
relation between inputs to get the output without explicitly knowing the equation.
Here the relation between inputs is (a+b)*3

for example output a=15,b=2 so (a+b)*3= 51

You can tweak the relation between inputs and it will still give good results.

"""


from numpy import exp,array,random,dot

class neural_network:
    def __init__(self):
        #always output same random numbers
        random.seed(1)
        
        #single neuron,with 2 inputs and 1 output and assign random weight
        self.weights=2*random.random((2,1))-1
        print("initial random weights: ",self.weights)
        
    def train(self,inputs,outputs,num):
        for iteration in range(num):
            #print("input: ",input)
            output=self.think(inputs)
            print("output for these inputs :",output)
            #print("original output: ",outputs)
            error= outputs- output
            print("calculated error: ",error)
            #backpropogation
            adjustment=0.01*dot(inputs.T,error)
            print("adjustment in weights: ",adjustment)
            self.weights+=adjustment
            print("weight after ",iteration," iteration: ",self.weights)
            
    def think(self, inputs):
        return (dot(inputs,self.weights))
    
neural_network= neural_network()

#the Training set
inputs= array([[2,3],[1,1],[5,2],[12,3]])
outputs=array([[15,6,21,45]]).T

#training 
neural_network.train(inputs,outputs,1000)

print(neural_network.think(array([15,2])))