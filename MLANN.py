import random
import math

from NN import matrix
from NN.matrix import *

from NN import sigmoid
from NN.sigmoid import *

class MLP:
  def __init__(self, input_size, hidden_size, output_size):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    
    self.weights_ih = create_matrix(input_size, hidden_size)
    for r in range(input_size):
      for c in range(hidden_size):
        self.weights_ih[r][c] = random.uniform(-1, 1)
    
    self.bias_h = create_matrix(1, hidden_size)
    for c in range(hidden_size):
      self.bias_h[0][c] = random.uniform(-1, 1)
      
    self.weights_ho = create_matrix(hidden_size, output_size)
    for r in range(hidden_size):
      for c in range(output_size):
        self.weights_ho[r][c] = random.uniform(-1, 1)
    
    self.bias_o = create_matrix(1, output_size)
    for c in range(output_size):
      self.bias_o[0][c] = random.uniform(-1, 1)
  
  def forward(self, inputs):
    inputs_matrix = [[x for x in inputs]]
    
    hidden_layer_input = dot(inputs_matrix, self.weights_ih)
    hidden_layer_input = add_matrices(hidden_layer_input, self.bias_h)
    self.hidden_layer_output = apply_function(hidden_layer_input, sigmoid)
    
    output_layer_input = dot(self.hidden_layer_output, self.weights_ho)
    output_layer_input = add_matrices(output_layer_input, self.bias_o)
    self.predicted_output = apply_function(output_layer_input, sigmoid)
    
    return self.predicted_output[0]
  
  def train(self, training_inputs, training_outputs, num_epochs=10000, learning_rate=0.5):
    for epoch in range(num_epochs):
      total_error = 0.0
      for i in range(len(training_inputs)):
        inputs = training_inputs[i]
        expected_output = [[x for x in training_outputs[i]]]
        
        #forward pass
        predicted_output_list = self.forward(inputs)
        predicted_output_matrix = [[x for x in predicted_output_list]]#convert to matrix
        
        #back prop
        
        #1 output layer error
        output_error = subtract_matrices(expected_output, predicted_output_matrix)
        #print("output error: %f" % output_error)
        total_error += sum(abs(e) for row in output_error for e in row)
        
        d_predicted_output = apply_function(predicted_output_matrix, sigmoid_derivative)
        output_delta = multiply_matrices_elementwise(output_error, d_predicted_output)
        
        #2 hidden to out weights and biases update
        weights_ho_gradient = dot(transpose(self.hidden_layer_output), output_delta)
        
        self.weights_ho = add_matrices(self.weights_ho, multiply_scalar_matrix(weights_ho_gradient, learning_rate))
        self.bias_o = add_matrices(self.bias_o, multiply_scalar_matrix(output_delta, learning_rate))
        
        #3 hidden layer error
        hidden_layer_error = dot(output_delta, transpose(self.weights_ho))
        
        d_hidden_output = apply_function(self.hidden_layer_output, sigmoid_derivative)
        hidden_delta = multiply_matrices_elementwise(hidden_layer_error, d_hidden_output)
        
        #4 input to hidden weights and biases update
        inputs_matrix_for_gradient = [[x] for x in inputs]
        weights_ih_gradient = dot(inputs_matrix_for_gradient, hidden_delta)
        
        self.weights_ih = add_matrices(self.weights_ih, multiply_scalar_matrix(weights_ih_gradient, learning_rate))
        self.bias_h = add_matrices(self.bias_h, multiply_scalar_matrix(hidden_delta, learning_rate))
        
      if epoch % 1000 == 0:
        print("Epoch: ", str(epoch))
        print("Total Error: ", str(total_error))