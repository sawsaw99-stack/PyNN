import math

#sigmoid activation function
def sigmoid(x):
  try:
    return 1 / (1 + math.exp(-x))
  except OverflowError:
    return 0.0

#derivative of th sigmoid func
def sigmoid_derivative(x):
  return x * (1 - x)
