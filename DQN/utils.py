# Useful Tensorflow & DQN utilities

import tensorflow as tf

# Copying weights from one network to another one
def copy_weights_from_one_nn_to_other(source_network, target_network):
    source_variables = source_network.session.run(source_network.variables)
    previous_target_variables = target_network.session.run(target_network.variables)

    for i in range(len(source_variables)):
        target_network.session.run(target_network.variables[i].assign(source_variables[i]))

    # These are used to see that everything transferred okay.
    #source_variables = source_network.session.run(source_network.variables)
    #target_variables = target_network.session.run(target_network.variables)
    #print("here")

def predict(model, observation):
    return model.session.run(model.output_layer, {model.input_layer : observation})

# Huber loss uses MSE (mean squared error) for low absolute values (<=1 in this case), and MAE (mean absolute error) for
# high absolute values (>1 in this case)
def get_huber_loss(a, b):
    error = a - b
    
    # These need to be used specifically to accommodate TensorFlow
    quadratic_term = error*error / 2
    return quadratic_term
    #linear_term = abs(error) - 1/2
    #use_linear_term = tf.cast(abs(error) > 1.0, tf.float32)
    
    #return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term