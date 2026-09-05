# Call libraries :
import numpy as np # compute numerical values                            
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # use API in high abstraction level   


# Initializing (hyper)parameters : 

# creating an optimizer with types of gradient descent methods.
# AdaGrad with epsilon :
AdaGrad = keras.optimizers.Adagrad( lr = 0.01 , epsilon = 1e-07 )

# RMSprop with rho and epsilon : 
RMSprop = keras.optimizers.RMSprop( lr = 0.001 , rho = 0.8 , epsilon = 1e-08 )

# Adam with epsilon and decay :
Adam = keras.optimizers.Adam( lr = 0.01 , epsilon = 0.1 , decay = 0.6 )


# Create a sequential neural network : 
neural_network = keras.Sequential( [ ] )


# Training neural network :

# creating compilers with type of loss function , type of optimizer , and type of supervised metric :
# compiler for AdaGrad : 
neural_network.compile( loss = 'mean_squared_error' , optimizer= 'AdaGrad' , metrics = [ 'accuracy' ] )
# compiler for RMSProp :
neural_network.compile( loss = 'mean_squared_error' , optimizer= 'RMSprop' , metrics = [ 'accuracy' ] )
# compiler for Adam :
neural_network.compile( loss = 'mean_squared_error' , optimizer= 'Adam' , metrics = [ 'accuracy' ] )