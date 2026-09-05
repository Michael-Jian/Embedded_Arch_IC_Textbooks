# Call libraries :
import numpy as np # compute numerical values     
import tensorflow as tf # choose specifc DL framework                  
keras = tf.keras  # use API in high abstraction level   


# Initializing (hyper)parameters : 
epoch = 14
batch_size = 64

# Load and Prepare training dataset and test dataset :
# load MNIST dataset
mnist = keras.datasets.mnist
( train_images, train_labels ) , ( test_images , test_labels ) = mnist.load_data()


# Create a sequential neural network : 
neural_network = keras.Sequential ( [
    # input layer : 
    # 'Flatten' suggest that it is a layer which reshapes multi-dimensional inputs into one dimensional inputs.
    keras.layers.Flatten( input_shape = ( 28 , 28 ) ) , 
    # hidden layer : 
    # activation function : ReLU fumnction.
    # initializer of input weights : He_Normal fuction.
    # initializer of bias weights  : zeros.
    keras.layers.Dense( 25 , activation='relu' , kernel_initializer='he_normal' , bias_initializer='zeros' ) ,
    # output layer : 
    # activation function : SoftMax fumnction.
    # initializer of input weights : Glorot_Uniform fuuction with uniform distribution.
    # initializer of bias weights  : zeros.
    keras.layers.Dense( 10 , activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros' )
                                    ] )


# Training neural network :
# creating a compiler with type of loss function , type of optimizer , and type of supervised metric :
# loss function : categorical cross entropy function.
# optimizer : Adam function ( use 'adam' directly without initializon due to no adjusted hyperparameters )
neural_network.compile( loss = 'categorical_crossentropy', optimizer = 'adam' , metrics= [ 'accuracy' ] )
# creating a trainer with training datasets , test datasets , epoch , batch size , type of verbosity , shuffle mechanism
neural_network_trainer = neural_network.fit( 
                         train_images , train_labels ,
                         validation_data = ( test_images , test_labels ) , 
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
