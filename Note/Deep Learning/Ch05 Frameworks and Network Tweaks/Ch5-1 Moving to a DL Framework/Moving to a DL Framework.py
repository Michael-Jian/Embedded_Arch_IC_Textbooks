# Call Libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level                           
to_categorical = tf.keras.utils.to_categorical # one-hot encoding target labels
import logging  # controlling runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppressing warning logs
tf.random.set_seed( 7 )     # initializing repeatable random sead


# Initializing (Hyper)parameters  : 
epoch = 20
batch_size = 1 # updating weights with batch size = 1 is equal to updating weights per pair training example.
# initializing weights with randomly uniform method wth specific range
RU_weight_initializer = keras.initializers.RandomUniform( minval = -0.1 , maxval = 0.1 )
# creating an optimizer with type of gradient descent method and value of learning rate.
SGD_optimizer = keras.optimizers.SGD( learning_rate = 0.01 )


# Load and Prepare Training Dataset and Test Dataset :
# load MNIST dataset ( 60000 training data and 10000 test data which have 784 features per data )
mnist = keras.datasets.mnist
( training_images, training_labels ) , ( test_images , test_labels ) = mnist.load_data()
#standarizing :
mean = np.mean( training_images )
stddev = np.std( training_images )
training_images = ( training_images - mean ) / stddev
training_images = ( test_images - mean ) / stddev
# transfer ground truth of training dataset and test dataset into one-hot coding with 10 classes
training_labels = to_categorical( training_labels , num_classes = 10 )
test_labels = to_categorical( test_labels , num_classes = 10 )


# Create a sequential Neural Network : 
neural_network = keras.Sequential( [ 
    # input layer : 
    # 'Flatten' suggest that it is a layer which reshapes multi-dimensional inputs into one dimensional inputs.
    keras.layers.Flatten( input_shape = ( 28 , 28 ) ) , 
    # hidden layer : 
    # 'Dense' suggest that it's a fully connected layer.
    # creating the layer with num of neuron , type of activation funct , 
    # initialzing input weights , initialzing bias weights 
    # cannot write ' 0 ' but only 'zeros' for specific tensor. 
    keras.layers.Dense( 25 , activation = 'tanh' , kernel_initializer = RU_weight_initializer , bias_initializer = 'zeros' ) ,
    # output layer : 
    # 'Dense' suggest that it's a fully connected layer.
    # creating the layer with num of neuron , type of activation funct , 
    # initialzing input weights , initialzing bias weights
    # cannot write ' 0 ' but only 'zeros' for specific tensor. 
    keras.layers.Dense( 10 , activation='sigmoid' , kernel_initializer = RU_weight_initializer , bias_initializer = 'zeros' ) 
                                    ] )


# Training neural network :
# creating a compiler with type of loss function , type of optimizer , and type of supervised metric.
neural_network.compile( loss = 'mean_squared_error' , optimizer = SGD_optimizer , metrics = [ 'accuracy' ] )
# creating a trainer with training datasets , test datasets , epoch , batch size , type of verbosity , shuffle mechanism
neural_network_trainer = neural_network.fit( 
                         training_images , training_labels ,
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
