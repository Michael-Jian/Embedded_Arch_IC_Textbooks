# Call Libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # using API in high abstraction level   
from keras import Sequential # use the Sequential Neural Network from keras
from keras.layers import Dense # use the Dense layer from kersas
from keras.regularizers import l2 # use L2 Regularization
from keras.layers import Dropout # use Iverted Drpout
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs


# Initializing (Hyper)parameters : 
epoch = 500 
batch_size = 16
model = Sequential() # initialize an empty sequential neural network.


# Load and Prepare Training Dataset and Test Dataset :
# load Boston Housing dataset  ( 404 training data and 102 test and 10000 test data which have 13 features per data ) 
boston_housing = keras.datasets.boston_housing 
( training_raw , training_labels ), ( test_raw , test_labels ) = boston_housing.load_data()
#standarizing :
# x_mean = np.mean( training_raw , axis = 0 ) :
# axis = 0 : create an array containing the individual mean of each of the 13 features respectively , 404 x 1 values per feature.
# this ensures each feature is standardized independently based on its own distribution,
# preventing small-scale features from being dominated by large-scale features.
# x_mean = np.mean( training_raw ) :
# without axis : create a single scalar mean of all 13 features at once , 404 x 13 values.
# because 13 features' scales and units vary drastically ,
# the overall mean is dominated by large-magnitude features, making standardization statistically meaningless.
raw_mean = np.mean( training_raw , axis = 0 ) 
raw_stddev = np.std( training_raw , axis = 0 ) 
training_raw = ( training_raw - raw_mean ) / raw_stddev
test_raw = ( test_raw - raw_mean ) / raw_stddev 


# Create a sequential Neural Network : 
# input layer + first hidden layer : 
# 'Dense' suggest that it's a fully connected layer.
# creating the layer with num of neuron , type of activation funct , 
# regularizer for input weights , regularizer for bias weights 
model.add( Dense ( 64 , activation = 'relu' , 
                   kernel_regularizer=l2( 1e-05 ) , # Lambda = 1 x 1e-05 in L2 Weight Decay.
                   bias_regularizer=l2( 0 ) , # Lambda = 0 in L2 Weight Decay because bias weights are not regularized typically. 
                   input_shape=[ 13 ] # input 13 features per data.
                  )
         ) 
model.add( Dropout( 0.2 ) ) # add Dropout with dropout rate = 20% in the first hidden layer.
# second hidden layer : 
# 'Dense' suggest that it's a fully connected layer.
# creating the layer with num of neuron , type of activation funct , 
# regularizer for input weights , regularizer for bias weights 
model.add( Dense ( 64 , activation = 'relu' , 
                   kernel_regularizer=l2( 1e-05 ) , # Lambda = 1 x 1e-05 in L2 Weight Decay.
                   bias_regularizer=l2( 0 ) # Lambda = 0 in L2 Weight Decay because bias weights are not regularized typically. 
                  )
         )
# output layer : 
# 'Dense' suggest that it's a fully connected layer.
# creating the layer with num of neuron , type of activation funct , 
# regularizer for input weights , regularizer for bias weights 
model.add( Dropout ( 0.2 ) ) # add Dropout with dropout rate = 20% in the first hidden layer.
model.add(Dense( 1 , activation = 'linear' , 
                 kernel_regularizer=l2( 1e-05 ) , # Lambda = 1 x 1e-05 in L2 Weight Decay.
                 bias_regularizer=l2( 0 ) # Lambda = 0 in L2 Weight Decay because bias weights are not regularized typically. 
               )
         )
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

# Training neural network :
# creating a compiler with type of loss function , type of optimizer , and type of supervised metric.
model.compile( loss ='mean_squared_error', optimizer = 'adam', metrics = [ 'mean_absolute_error' ] ) 
# creating a trainer with training datasets , test datasets , epoch , batch size , type of verbosity , shuffle mechanism
model_trainer = model.fit( training_raw , training_labels , 
                           validation_data=( test_raw , test_labels ) , 
                           epochs = epoch , 
                           batch_size = batch_size , 
                           # creating a verbose mode for training progress output.
                           # verbosity = 0 : Silent ( no log for training progress )
                           # verbosity = 1 : Progress bar ( interactive logs for training progress each batch )
                           # verbosity = 2 : One line per epoch (cleaner logs for training progress each epoch )
                           verbose = 2 , 
                           # creating a shuffle mechanism whether randomly permute the training data at the beginning of each epoch. 
                           shuffle = True 
                         )

# Print Out Results :
# evaluate the predicted output of the test dataset with the desired output of the test dataset ( ex : test_labels ) :
predicted_output = model.predict( test_raw )
for i in range( 0 , 4 ) :
    print(' Predicted Output ' , i , ' : ', predicted_output[ i ] , 'vs. Ground Truth : ', test_labels[ i ] ) 
    
    
    


