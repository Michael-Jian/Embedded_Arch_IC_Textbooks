# Call libraries :
import numpy as np # compute numerical values                      
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # use API in high abstraction level    


# Hidden layer :
keras.layers.Dense( 64 , activation = 'tanh' ) , 
# applying batch normalization " after " activation function.
keras.layers.BatchNormalization() ,