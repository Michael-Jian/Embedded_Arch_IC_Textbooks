# Call libraries :
import numpy as np # compute numerical values                          
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # use API in high abstraction level    


# Hidden layer :
keras.layers.Dense( 64 ) ,
# applying batch normalization " before " activation function.
keras.layers.BatchNormalization() ,
keras.layers.Activation( 'tanh' ) ,


