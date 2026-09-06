# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose specifc DL framework
keras = tf.keras  # use API in high abstraction level        
import matplotlib.pyplot as plt  # for protraiting figures
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs


# Load and Prepare Training Dataset and Test Dataset :
# load CIFAR dataset ( 50000 training data and 10000 test data which have 32 x 32 x 3 features per data )
cifar_dataset = keras.datasets.cifar10 
( training_images , training_labels ), ( test_images , test_labels ) = cifar_dataset.load_data()


# Print Out Results :
print( 'Training Label :' , 'category' , training_labels[ 100 ] ) 
print( 'Training Image :' , "cifar_index_100.png" )
plt.figure( figsize = ( 5 , 5 ) ) # create a figure with size of 5 x 5 ( inch^2 )
plt.imshow( training_images [ 100 ] ) # insert the training image [ 100 ] into the figure
# plt.show() # show the figure on the screen
# save the  figure into .png format
plt.savefig( 'cifar_index_100.png' ) 

    