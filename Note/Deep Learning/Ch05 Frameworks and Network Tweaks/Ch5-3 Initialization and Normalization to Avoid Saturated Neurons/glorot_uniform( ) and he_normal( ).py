# Call libraries :
import numpy as np # compute numerical values                          
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # use API in high abstraction level 


# Create a sequential neural network : 
neural_network = keras.Sequential([
   
    keras.layers.Flatten( input_shape = ( 28 , 28 ) ) , 
   
    # applying glorot uniform function for weights initialization in the hidden layer.
    keras.layers.Dense( 25 , activation = 'tanh' , kernel_initializer = 'glorot_uniform' , bias_initializer = 'zeros' ) ,
    # applying glorot he normal function for weights initialization in the hidden layer.
    keras.layers.Dense( 25 , activation = 'relu' , kernel_initializer = 'he_normal' , bias_initializer = 'zeros' ) ,
    
    keras.layers.Dense( 10 , activation='sigmoid' , kernel_initializer='glorot_uniform' , bias_initializer='zeros')
                         ] )