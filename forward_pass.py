import numpy as np

# Define the linear function
# The function should take inputs, weights, and a bias, and return the weighted sum
def linear(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

#Define the step function
# The step function returns 1 if the input z is greater than or equal to 0, otherwise it returns 0
def step(z):
    return 1 if z >= 0 else 0

# Write the perceptron forward pass function
# This function will take an array of inputs, a weight vector, and a bias, and return the prediction for each input
def perceptron_forward_pass(input_set, weights, bias):
    pred = []
    for x in input_set:
        z = linear(x, weights, bias)
        pred.append(step(z))
    return pred

# Test the perceptron with sample inputs and weights
# Example input and weight vectors
input_set = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 2-input binary combinations
weights = np.array([0.5, 0.5])                           # Example weights
bias = -0.7                                              # Example bias

pred = perceptron_forward_pass(input_set, weights, bias)
print("Predictions:", pred)   # [0, 0, 0, 1]

# Experiment with other values for the weights and bias
weights = np.array([1.0, 1.0]) # Modify the weights
bias = -0.5   # Modify the bias

pred = perceptron_forward_pass(input_set, weights, bias)
print("New predictions:", pred)   # [0, 1, 1, 1]

# Define the labels for the input set
# Labels for an AND gate (True outputs for the input combinations)
labels = np.array([0, 0, 0, 1])