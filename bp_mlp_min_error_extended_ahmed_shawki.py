import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mlp_shawki_extended(max_err, x, target, learnrate, max_iterations=10000):
    # Updated weight initialization for a larger network
    # Input to first hidden layer (4 nodes)
    weights_input_hidden1 = np.array([[0.5, -0.6, 0.2, -0.1],
                                      [0.1, -0.2, 0.4, 0.3],
                                      [0.1, 0.7, -0.2, 0.5]])

    # First hidden layer to second hidden layer (3 nodes)
    weights_hidden1_hidden2 = np.array([[0.3, -0.5, 0.7],
                                        [0.1, 0.2, -0.3],
                                        [-0.2, 0.6, -0.1],
                                        [0.4, -0.2, 0.1]])

    # Second hidden layer to output (1 node)
    weights_hidden2_output = np.array([0.2, -0.1, 0.5])

    error = 100
    i = 0
    while abs(error) > max_err and i < max_iterations:
        i += 1

        ## Forward pass
        # Input to first hidden layer
        hidden_layer1_input = np.dot(x, weights_input_hidden1)
        hidden_layer1_output = sigmoid(hidden_layer1_input)

        # First hidden layer to second hidden layer
        hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
        hidden_layer2_output = sigmoid(hidden_layer2_input)

        # Second hidden layer to output
        output_layer_in = np.dot(hidden_layer2_output, weights_hidden2_output)
        output = sigmoid(output_layer_in)

        ## Backward pass
        error = target - output

        # Output layer error term
        output_error_term = error * output * (1 - output)

        # Second hidden layer error term
        hidden2_error_term = np.dot(output_error_term, weights_hidden2_output) * \
                             hidden_layer2_output * (1 - hidden_layer2_output)

        # First hidden layer error term
        hidden1_error_term = np.dot(hidden2_error_term, weights_hidden1_hidden2.T) * \
                             hidden_layer1_output * (1 - hidden_layer1_output)

        # Weight updates for second hidden layer to output
        delta_w_h2_o = learnrate * output_error_term * hidden_layer2_output

        # Weight updates for first hidden layer to second hidden layer
        delta_w_h1_h2 = learnrate * np.outer(hidden_layer1_output, hidden2_error_term)

        # Weight updates for input to first hidden layer
        delta_w_i_h1 = learnrate * np.outer(x, hidden1_error_term)

        # Update weights
        weights_hidden2_output += delta_w_h2_o
        weights_hidden1_hidden2 += delta_w_h1_h2
        weights_input_hidden1 += delta_w_i_h1

    return output, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output


# Testing the updated MLP with 2 hidden layers and more nodes
x = np.array([0.5, 0.1, -0.2])  # Input array
target = 0.2  # Target value
learnrate = 0.5  # Learning rate

print("*****************************")
output, w_i_h1, w_h1_h2, w_h2_o = mlp_shawki_extended(0.001, x, target, learnrate)

print(f"Input was {x}")
print(f"Target was {target}, and final output is {output}, error = {target - output}")
print('Final weights input to first hidden layer: ', w_i_h1)
print('Final weights first hidden to second hidden layer: ', w_h1_h2)
print('Final weights second hidden to output: ', w_h2_o)
print("*****************************")
