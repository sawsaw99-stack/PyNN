"""
A simple, single layer Artificial Neural Network.

Works on MicroPython! (Although, I recommend pre-loading the weights and biases first)

"""

import random
from NN import sigmoid
import math

class ArtificialNeuron:
  def __init__(self):
    #seed for repr random weights (for testing)
    random.seed(1)
    
    #init weights randomly (eg between -1 and 1)
    #for a single neuron w/ 3 inputs
    self.weights = [random.uniform(-1, 1) for _ in range(3)]
    self.bias = random.uniform(-1, 1) #optional bias
    
  def forward(self, inputs):
    #calc weighted sum of inputs
    weighted_sum = sum(input_val * weight for input_val, weight in zip(inputs, self.weights)) + self.bias
    #apply activation func
    output = sigmoid.sigmoid(weighted_sum)
    return output
  
  def train(self, training_inputs, training_outputs, num_epochs, learning_rate):
    for epoch in range(num_epochs):
      total_error = 0
      for inputs, expected_output in zip(training_inputs, training_outputs):
        #forward pass
        predicted_output = self.forward(inputs)
        
        #calculate the error
        error = expected_output - predicted_output
        total_error += abs(error) #for mon
        
        #calc the delta (how much to adj weights)
        #this uses derivative of the sigmoid
        d_predicted_output = error * sigmoid.sigmoid_derivative(predicted_output)
        
        #adjust weights
        for i in range(len(self.weights)):
          self.weights[i] += learning_rate * d_predicted_output * inputs[i]
        #adj bias
        self.bias += learning_rate * d_predicted_output

# WIP
"""
class FeedbackANN(ArtificialNeuron):
  def __init__(self):
    SimpleNeuralNetwork.__init__(self)
    
    self.ins = []
    self.outs = []
  
  def train(self, ins, outs, epochs, learn):
    SimpleNeuralNetwork.train(self, ins, outs, epochs, learn)
    
    for x in ins:
      if x in self.ins:
        pass
      else:
        self.ins.append(x)
    
    for x in outs:
      if x in self.outs:
        pass
      else:
        self.outs.append(x)

  def forward_feedback(self, tin):
    stuff = SimpleNeuralNetwork.forward(self, tin)
    
    return stuff
    
    print("How did AI do?")
    r = input()
    
    if tin in self.ins:
      n = self.ins[self.insindex]
"""
