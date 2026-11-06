import numpy as np

# Define the linear function
def linear(inputs, weights, bias):
    return np.dot(inputs, weights) + bias