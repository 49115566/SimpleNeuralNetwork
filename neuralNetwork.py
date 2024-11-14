import numpy as np
from typing import List

class Neuron:
    def __init__(self, num_inputs: int, activation: str = 'relu', seed: int = None):
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility
        limit = np.sqrt(6 / (num_inputs + 1))  # Xavier initialization limit
        self.weights = np.random.uniform(-limit, limit, num_inputs)  # Initialize weights
        self.bias = np.random.uniform(-limit, limit)  # Initialize bias
        # Set activation function and its derivative
        if activation == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative
        elif activation == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'linear':
            self.activation_function = self.linear
            self.activation_derivative = self.linear_derivative
        else:
            raise ValueError("Unsupported activation function")

    def activate(self, inputs: np.ndarray) -> float:
        self.inputs = inputs  # Store inputs for backpropagation
        self.z = np.dot(inputs, self.weights) + self.bias  # Calculate weighted sum
        self.output = self.activation_function(self.z)  # Apply activation function
        return self.output

    def sigmoid(self, x: float) -> float:
        x = np.clip(x, -500, 500)  # Clip input values to avoid overflow in exp
        return 1 / (1 + np.exp(-x))  # Sigmoid activation function

    def sigmoid_derivative(self, sigmoid_output: float) -> float:
        return sigmoid_output * (1 - sigmoid_output)  # Derivative of sigmoid

    def relu(self, x: float) -> float:
        return np.maximum(0, x)  # ReLU activation function

    def relu_derivative(self, x: float) -> float:
        return np.where(x > 0, 1, 0)  # Derivative of ReLU

    def linear(self, x: float) -> float:
        return x  # Linear activation function

    def linear_derivative(self, x: float) -> float:
        return 1  # Derivative of linear

class NeuralNetwork:
    def __init__(self, num_inputs: int, num_hidden_neurons1: int, num_hidden_neurons2: int, num_output_neurons: int, seed: int = None):
        # Initialize neurons for each layer
        self.hidden_neurons1 = [Neuron(num_inputs, activation='relu', seed=seed) for _ in range(num_hidden_neurons1)]
        self.hidden_neurons2 = [Neuron(num_hidden_neurons1, activation='relu', seed=seed) for _ in range(num_hidden_neurons2)]
        self.output_neurons = [Neuron(num_hidden_neurons2, activation='linear', seed=seed) for _ in range(num_output_neurons)]

    def feedforward(self, inputs: np.ndarray) -> List[float]:
        # Forward pass through the first hidden layer
        hidden_outputs1 = [neuron.activate(inputs) for neuron in self.hidden_neurons1]
        # Forward pass through the second hidden layer
        hidden_outputs2 = [neuron.activate(np.array(hidden_outputs1)) for neuron in self.hidden_neurons2]
        # Forward pass through the output layer
        final_outputs = [neuron.activate(np.array(hidden_outputs2)) for neuron in self.output_neurons]
        return final_outputs

    def train(self, inputs: np.ndarray, targets: np.ndarray, learning_rate: float, epochs: int, validation_data=None, patience=10):
        best_loss = float('inf')  # Initialize best loss for early stopping
        patience_counter = 0  # Initialize patience counter for early stopping

        for epoch in range(epochs):
            for x, y in zip(inputs, targets):
                # Forward pass
                hidden_outputs1 = [neuron.activate(x) for neuron in self.hidden_neurons1]
                hidden_outputs2 = [neuron.activate(np.array(hidden_outputs1)) for neuron in self.hidden_neurons2]
                outputs = [neuron.activate(np.array(hidden_outputs2)) for neuron in self.output_neurons]
                
                print(f'Epoch {epoch + 1}/{epochs}, Input: {x}, Target: {y}, Output: {outputs}')

                # print(f'output_neurons:')
                # Backward pass for output layer
                for i, neuron in enumerate(self.output_neurons):
                    error = outputs[i] - y[i]
                    d_loss_d_output = 2 * error
                    d_output_d_z = neuron.activation_derivative(neuron.output)
                    d_z_d_weights = neuron.inputs
                    d_z_d_bias = 1

                    # Gradient clipping
                    gradient_weights = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_weights
                    gradient_bias = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_bias
                    gradient_weights = np.clip(gradient_weights, -1, 1)
                    gradient_bias = np.clip(gradient_bias, -1, 1)

                    # Update weights and biases
                    neuron.weights -= gradient_weights
                    neuron.bias -= gradient_bias

                # Backward pass for second hidden layer
                for i, neuron in enumerate(self.hidden_neurons2):
                    error = sum([output_neuron.weights[i] * (2 * (output_neuron.output - y[j]) * output_neuron.activation_derivative(output_neuron.output)) for j, output_neuron in enumerate(self.output_neurons)])
                    d_loss_d_output = error
                    d_output_d_z = neuron.activation_derivative(neuron.z)
                    d_z_d_weights = neuron.inputs
                    d_z_d_bias = 1

                    # Gradient clipping
                    gradient_weights = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_weights
                    gradient_bias = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_bias
                    gradient_weights = np.clip(gradient_weights, -1, 1)
                    gradient_bias = np.clip(gradient_bias, -1, 1)

                    # Update weights and biases
                    neuron.weights -= gradient_weights
                    neuron.bias -= gradient_bias

                # Backward pass for first hidden layer
                for i, neuron in enumerate(self.hidden_neurons1):
                    error = sum([hidden_neuron.weights[i] * hidden_neuron.activation_derivative(hidden_neuron.z) * sum([output_neuron.weights[j] * (2 * (output_neuron.output - y[k]) * output_neuron.activation_derivative(output_neuron.output)) for k, output_neuron in enumerate(self.output_neurons)]) for j, hidden_neuron in enumerate(self.hidden_neurons2)])
                    d_loss_d_output = error
                    d_output_d_z = neuron.activation_derivative(neuron.z)
                    d_z_d_weights = neuron.inputs
                    d_z_d_bias = 1

                    # Gradient clipping
                    gradient_weights = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_weights
                    gradient_bias = learning_rate * d_loss_d_output * d_output_d_z * d_z_d_bias
                    gradient_weights = np.clip(gradient_weights, -1, 1)
                    gradient_bias = np.clip(gradient_bias, -1, 1)

                    # Update weights and biases
                    neuron.weights -= gradient_weights
                    neuron.bias -= gradient_bias

                    # print(f'error: {error}, Δw: {learning_rate * d_loss_d_output * d_output_d_z * d_z_d_weights}, Δb: {learning_rate * d_loss_d_output * d_output_d_z * d_z_d_bias}')

# Example usage:
# Create a neural network with 3 input neurons, 5 hidden neurons in each of the 2 hidden layers, and 3 output neurons
network = NeuralNetwork(num_inputs=1, num_hidden_neurons1=64, num_hidden_neurons2=64, num_output_neurons=1, seed=42)

# Training data (inputs and targets)
inputs = np.array([[2.0], [1.0], [-1.0], [100.0], [-10.0], [13.0], [-4.0], [-16.0], [-18.0], [19], [27], [22], [-6], [-1], [3]])
targets = inputs ** 10

mean = np.mean(inputs, axis=0)
std = np.std(inputs, axis=0)

# Normalize inputs
inputs = (inputs - mean) / std

# Train the network
network.train(inputs, targets, learning_rate=0.0001, epochs=1000)

# Test the network with different inputs
test_inputs = np.array([[2.0], [1.0], [-1.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
test_inputs = (test_inputs - mean) / std

for test_input in test_inputs:
    outputs = network.feedforward(test_input)
    print(f'Input: {test_input}, Output: {outputs}')

print("Using training dataset:")
for input in inputs:
    print(f'Input: {input}, Output: {network.feedforward(input)}')