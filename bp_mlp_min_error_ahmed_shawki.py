import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def mlp_shawki(max_err,x,target,learnrate,max_iterations=10000):
    
    weights_hidden_output=np.array([0.1, -0.3])
    weights_input_hidden=np.array([[0.5, -0.6],
                                    [0.1, -0.2],
                                    [0.1, 0.7]])
    error=100
    i=0
    
    while abs(error) > max_err and i < max_iterations:
        i+=1
        ## Forward pass
        hidden_layer_input = np.dot(x, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
        output = sigmoid(output_layer_in)
        #print(f'Iteration {i+1} - Output: {output}')
        ## Backwards pass
        ## TODO: Calculate output error
        error = target - output
        #print('error',error)

        # TODO: Calculate error term for output layer
        output_error_term = error * output * (1 - output)
        #print('output_error_term',output_error_term)

        # TODO: Calculate error term for hidden layer
        hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                            hidden_layer_output * (1 - hidden_layer_output)

        # TODO: Calculate change in weights for hidden layer to output layer
        delta_w_h_o = learnrate * output_error_term * hidden_layer_output

        # TODO: Calculate change in weights for input layer to hidden layer
        delta_w_i_h = learnrate * hidden_error_term * x[:, None]

        #print('Change in weights for hidden layer to output layer:')
        #print(delta_w_h_o)
        #print('Change in weights for input layer to hidden layer:')
        #print(delta_w_i_h)

        # Update weights
        weights_hidden_output += delta_w_h_o
        weights_input_hidden += delta_w_i_h  
    return output, weights_input_hidden,weights_hidden_output

x = np.array([0.5, 0.1, -0.2])
target = 0.2
learnrate = 0.5

print("*****************************")       
output,w_i_h,w_h_o=mlp_shawki(0.001,x,target,learnrate)

print(f"input was  {x} ")
print(f"target was :{target} ,and final output is{output},error={target - output} ")
print('final weights_input_hidden: ', w_i_h)
print('final weights_hidden_output: ', w_h_o)
print("*****************************")