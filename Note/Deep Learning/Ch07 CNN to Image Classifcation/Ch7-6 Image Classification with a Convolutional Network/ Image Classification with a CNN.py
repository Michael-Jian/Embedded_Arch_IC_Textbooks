# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose the specifc DL framework
keras = tf.keras  # use the API in high abstraction level   
from keras.utils import to_categorical # use a one hot coding utensil
from keras.models import Sequential # use a Sequential Neural Network from keras
from keras.layers import Conv2D # use a convolutional layer from kersas
from keras.layers import Flatten # use a 1D flat layer from kersas
from keras.layers import Dense # use a fully connected layer from kersas
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs


# Initializing (Hyper)parameters : 
epoch = 128
batch_size = 32
model = Sequential() # initialize an empty sequential neural network.

# Load and Prepare Training Dataset and Test Dataset :
# load CIFAR dataset ( 50000 training data and 10000 test data which have 32 x 32 x 3 features per data )
cifar_dataset = keras.datasets.cifar10 
( training_images , training_labels ), ( test_images , test_labels ) = cifar_dataset.load_data()

# standardize images : 
mean = np.mean( training_images ) 
stddev = np.std( training_images ) 
training_images = ( training_images - mean ) / stddev
test_images = ( test_images - mean ) / stddev
print( ' mean : ' , mean ) 
print('stdandard dieviation : ' , stddev ) 

# transfer lables into one hot code : 
# to_categorical( the original data , clsses of the original data ) 
train_labels = to_categorical( training_labels , num_classes = 10 )
test_labels = to_categorical( test_labels , num_classes = 10 ) 



# Create a sequential Neural Network : 
# input layer + first hidden layer : 
# 'Conv2D' suggest that it's a convolutional layer.
# padding = 'same' : padding function will count the number of extra values needed for padding and set the extra values as 0s  
# creating the layer with num of neuron , kernel size , length of stride , type of activation funct , type of padding mechanism
# creating the layer with regularizer for input weights , regularizer for bias weights 
# creating the layer with data inputs
model.add( Conv2D( 64, ( 5 , 5 ) , strides = ( 2 , 2 ) , activation = 'relu', padding = 'same' , 
                   kernel_initializer='he_normal', bias_initializer='zeros' , 
                   input_shape=( 32 , 32 , 3 ) ) ) # input 32 x 32 x 3 features per data.

# second hidden layer : 
# 'Conv2D' suggest that it's a convolutional layer.
# padding = 'same' : padding function will count the number of extra values needed for padding and set the extra values as 0s  
# creating the layer with num of neuron , kernel size , length of stride , type of activation funct , type of padding mechanism
# creating the layer with regularizer for input weights , regularizer for bias weights 
model.add( Conv2D( 64 , ( 3 , 3 ) , strides = ( 2 , 2 ) , activation = 'relu' , padding = 'same', 
                   kernel_initializer = 'he_normal' , bias_initializer = 'zeros' ) )

# third hidden layer : 
# 'Flatten' suggest that it is a layer which reshapes multi-dimensional inputs into one dimensional inputs.
model.add( Flatten( ) ) # reshape the outputs from the convolutional layer above for the usage of the following fully connected layer

# output layer : 
# 'Dense' suggest that it's a fully connected layer.
# creating the layer with num of neuron , type of activation funct , 
# creating the layer with regularizer for input weights , regularizer for bias weights 
model.add( Dense ( 10 , activation = 'softmax', 
                   kernel_initializer = 'glorot_uniform' , bias_initializer = 'zeros' ) )

# model.summary() :
# report network architecture regarding 
# 1. topology in a given layer.
# 2. output shapes in a given layer : ( batch size , output dimension ) 
# 3. parameter counts (  total number of trainable weights and biases ) in a given layer : 
# parameter counts = ( num of input weights per neuron * num of neurons ) + ( num of bias weights per neuron * num of neurons )
# 4. total parameter counts
# 5. total trainable parameter counts
# 6. total non-trainable parameter counts
model.summary() 

# Training Neural Network :
# creating a compiler with type of loss function , type of optimizer , and type of supervised metric.
model.compile( loss = 'categorical_crossentropy', optimizer = 'adam', metrics = [ 'accuracy'] ) 
# creating a trainer with training datasets , test datasets 
# creating a trainer with epoch , batch size 
# creating a trainer with type of verbosity 
# creating a trainer with shuffle mechanism
model_trainer = model.fit( training_images , training_labels , validation_data =( test_images , test_labels ) ,
                            epochs = epoch , batch_size = batch_size , 
                            # creating a verbose mode for training progress output.
                            # verbosity = 0 : Silent ( no log for training progress )
                            # verbosity = 1 : Progress bar ( interactive logs for training progress each batch )
                            # verbosity = 2 : One line per epoch (cleaner logs for training progress each epoch )
                            verbose = 2 , 
                            # creating a shuffle mechanism whether randomly permute the training data at the beginning of each epoch. 
                            shuffle=True
                            ) 

# Print Out Results :
# model.evaluate( training datasets / test datasets ) :
# the results of evaluation is despendable on the hyperparameters set on the model.compile().
# for example , model.evaluate( training datasets / test datasets ) will outputs 'loss' we set and 'metrics' we set 
# due to model.compile( loss , optimizer , metrics ) 
training_loss , training_acc = model.evaluate( training_images , training_labels )
test_loss , test_acc = model.evaluate( test_images , test_labels )
print( f" Training Error : " ,  ( 1.0 - training_acc ) * 100 )
print( f" Test Error : "  , ( 1.0 - test_acc ) * 100 )