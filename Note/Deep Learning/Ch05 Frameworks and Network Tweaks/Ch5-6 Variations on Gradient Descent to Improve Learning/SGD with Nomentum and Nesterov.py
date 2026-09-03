# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level   


# Initializing (hyper)parameters : 

# creating an optimizer with type of gradient descent method.
# SGD Variant with Momentum and Nesterov :
SGD_Variant = keras.optimizers.SGD( lr = 0.01 , momentum = 0.6 , decay = 0.0 , nesterov = True )


# Create a sequential neural network : 
neural_network = keras.Sequential( [ ] )


# Training neural network :

# creating a compiler with type of loss function , type of optimizer , and type of supervised metric.
neural_network.compile(loss = 'mean_squared_error' , optimizer= SGD_Variant , metrics = [ 'accuracy' ] )