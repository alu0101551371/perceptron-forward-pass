import numpy as np

# Define the linear function
# The function should take inputs, weights, and a bias, and return the weighted sum
def linear(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

#Define the step function
# The step function returns 1 if the input z is greater than or equal to 0, otherwise it returns 0
def step(z):
    return 1 if z >= 0 else 0