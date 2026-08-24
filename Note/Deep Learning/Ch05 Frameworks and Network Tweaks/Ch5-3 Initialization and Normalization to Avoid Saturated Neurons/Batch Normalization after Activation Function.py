# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level    


# Hidden layer :
keras.layers.Dense( 64 , activation = 'tanh' ) , 
# applying batch normalization " after " activation function.
keras.layers.BatchNormalization() ,