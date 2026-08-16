# Call NumPy , matplotlib , idx2numpy libraries :  
import numpy as np # for matrix multtplication.
import matplotlib.pyplot as plt # for protraiting learning curve chart
import idx2numpy # for MNIST dataset


# Insert files path : 
TRAINING_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-6 Classifying Handwritten Digits/mnist dataset/train-images.idx3-ubyte'
TRAINING_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-6 Classifying Handwritten Digits/mnist dataset/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-6 Classifying Handwritten Digits/mnist dataset/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-6 Classifying Handwritten Digits/mnist dataset/t10k-labels.idx1-ubyte'


# Prepare pair training examples : 
np.random.seed( 7 )  # set up a random seed '7' to start this shuffle command.
learning_rate = 0.005
epoch = 20 

# read training dataset and test dataset
def read_mnist() : 
    # reformate the files
    # idx2numpy.convert_from_file( filename ) : convert specific format file into NumPy 3D array format to fit for matrix multiplication.
    training_images = idx2numpy.convert_from_file( TRAINING_IMAGE_FILENAME )
    training_labels = idx2numpy.convert_from_file( TRAINING_LABEL_FILENAME )
    test_images = idx2numpy.convert_from_file( TEST_IMAGE_FILENAME )
    test_labels = idx2numpy.convert_from_file( TEST_LABEL_FILENAME )
        
    # reformate the input data
    # filename.reshape() : transfer the dimension/tensor of the file into different dimension/tensor structure , but the original data.
    # filename.reshape( x , y , z ) = transfer the dimension/tensor of the file into 3 dimension with x in length , y in width , z in height.
    # transfer 3D training_inputs_x (60000 , 28 , 28 ) into 2D training_inputs_x ( 60000 , 784 )
    training_inputs_x = training_images.reshape( 60000 , 784 )
    # transfer 3D test_inputs_x ( 10000 , 28 , 28 ) into 2D training_inputs_x ( 10000 , 784 )
    test_inputs_x = test_images.reshape( 10000, 784 )
    
    # standardize the input data
    # np.mean( filename ) : compute the mean value of all elements in the file.
    mean = np.mean( training_inputs_x )
    # np.std( filename ) : compute the standard deviation of all elements in the file.
    stddev = np.std( training_inputs_x ) 
    training_inputs_x = ( training_inputs_x - mean ) / stddev 
    # to avoid infomation leakage, neural network cannot get the info about whole test dataset
    # ( ex : mean of whole test dataset or standard deviation of whole test dataset ) at the same time,
    # but only each of the independent pair test example in a time
    test_inputs_x = ( test_inputs_x - mean ) / stddev 
   
    # reformate output y into one-hot encoing
    # np.zeros() : create an array / a matrix / a tensor structure with default value of each elements is 0.
    # np.zeros( x ) : create a x array structure with default value of each elements is 0.
    # np.zeros( x , y ) = create a  x * y matrix with default value of each elements is 0.
    # np.zeros( ( 60000 , 10 ) )  = a 60000 * 10 matrix with default value of each elements is 0.
    training_outputs_y = np.zeros( ( 60000 , 10 ) ) 
    # np.zeros( ( 10000 , 10 ) )  = a 10000 * 10 matrix with default value of each elements is 0.
    test_outputs_y = np.zeros( ( 10000, 10 ) )
    # training_labels : 5 3 6 ......8，total 60000 elements :
    # i (index) = 0 1 2 ......59999
    # j (value) = 5 3 6 ......  8
    # reformat into one-hot encoing :
    # i (index)   coresponding one-hot encoing ( only value in cell of index j = 1 , value in surplus cell  = 0 )
    #    0          0 0 0 0 0 1 0 0 0 0 
    #    1          0 0 0 1 0 0 0 0 0 0
    #    2          0 0 0 0 0 0 1 0 0 0
    #    .               .
    #    .               .
    #    .               .
    #    .               .
    #    .               .
    #    .               .
    #  59999        0 0 0 0 0 0 0 0 1 0
    for i, j in enumerate( training_labels ) :
        training_outputs_y[ i ][ j ] = 1
    # test_labels : 1 0 4 ......6，total 10000 elements :
        # i (index) = 0 1 2 ......9999
        # j (value) = 1 0 4 ...... 6
        # reformat into one-hot encoing :
        # i (index)   coresponding one-hot encoing ( only value in cell of index j = 1 , value in surplus cell  = 0 )
        #    0          0 1 0 0 0 0 0 0 0 0 
        #    1          1 0 0 0 0 0 0 0 0 0
        #    2          0 0 0 0 1 0 0 0 0 0
        #    .               .
        #    .               .
        #    .               .
        #    .               .
        #    .               .
        #    .               .
        #   9999        0 0 0 0 0 0 1 0 0 0    
    for i, j in enumerate( test_labels ) : 
        test_outputs_y[ i ][ j ] = 1 
    
    return training_inputs_x, training_outputs_y, test_inputs_x, test_outputs_y 

# training_inputs_x , test_inputs_x , training_outputs_y , test_outputs_y aboved are 4 local scope variables in read_mnist(),
# so we need to create training_inputs_x, training_outputs_y, test_inputs_x, test_outputs_y belowed 
# as 4 global scope variables to match the return value from read_mnist() 
training_inputs_x, training_outputs_y, test_inputs_x, test_outputs_y = read_mnist()

# the inital sequence of index list for training examples.
# range( start , stop ) : creat a rule interval = [ start, stop - 1 ] , but not entity that rule.
# range( stop ) = # range( 0 , stop ) : creat a rule interval = [ 0 , stop - 1 ] , but not entity that rule.
# list( x ) : create x into a string.
# list( range( start , stop )) = create a string [ start , start+1 , start+2 ...... , stop-1 ].
# list( range( stop ) ) = create a string [ 0 , 1 , 2...... , stop-1 ].
index_list = list( range( len( training_inputs_x ) ) )  


# Create weights in a layer :
def weight_per_layer( num_of_perceptron , num_of_input_weight_per_perceptron ) : 
    # num_of_input_weight_per_perceptron + 1 : add 1 cell for bias weight ( w0 ).
    # default vale of bias weight : weight[ i ][ 0 ] = 0
    weights = np.zeros( ( num_of_perceptron , num_of_input_weight_per_perceptron + 1 ) ) 
    for i in range( num_of_perceptron ) : 
        # range( 1 ,... : leave index = 0 for bias weight ( w0 )
        for j in range( 1 , ( num_of_input_weight_per_perceptron + 1 ) ) :
            # np.random.uniform( x, y ) : create a randomly and uniformly distributed value between x and y.
            # default input weights = weights[ i ][ 1 ] ~ weights[ i ][ num_of_input_weight_per_perceptron ] = value between -1.0 and 1.0
            weights[ i ][ j ] = np.random.uniform( -0.1 , 0.1 ) 
    return weights

# the hidden_layer has 25 hidden perceptrons and 784 input weights + 1 bias weight per perceptron
hidden_layer_w = weight_per_layer( 25 , 784 )
# the hidden_layer has 25 outputs for 25 hidden perceptrons.
hidden_layer_y = np.zeros( 25 ) 
# the hidden_layer has 25 error for 25 hidden perceptrons.
hidden_layer_error = np.zeros( 25 )
# the output_layer has 10 output perceptrons and 25 input weights + 1 bias weight per perceptron
output_layer_w = weight_per_layer( 10 , 25 ) 
# the output_layer has 10 outputs for 10 output perceptrons.
output_layer_y = np.zeros( 10 ) 
# the output_layer has 10 error for 10 output perceptrons.
output_layer_error = np.zeros( 10 )


# Report progress on the learning process : 
chart_x = [] # an array for X-axis data（ number of epoch ）
chart_y_training_error = []  # an array for Y-axis data（ training error ）
chart_y_test_error = []  # an array for Y-axis data（ test error ）

def show_learning( epoch, acc_of_training_error, acc_of_test_error ):
    # tell Python that the "chart_x" , "chart_y_training_error" , "chart_y_test_error" I use later is the old array in global scope, 
    # not the new one I create in this local scope.
    global chart_x
    global chart_y_training_error
    global chart_y_test_error 
    print( 'Epoch : ' , epoch , 
           ' , Accuracy of training error : ' , '%6.4f' % acc_of_training_error ,
           ' , Accuracy of test error : ' , '%6.4f' % acc_of_test_error 
         ) 
    chart_x.append( epoch + 1 ) # congest epoch ( start from 1 ) from front position.
    chart_y_training_error.append( 1.0 - acc_of_training_error ) # congest training error ( = 1 - accuracy of training error ) from front position.
    chart_y_test_error.append( 1.0 - acc_of_test_error ) # congest test error ( = 1 - accuracy of training error ) from front position.

def plot_learning() :
    # plt.plot( x , y  , ' w ' , label = ' z ' ) : function from the Matplotlib library used to draw 2D line and data points on a graph.
    # plt.plot( x , y , ' w ' , label = ' z ' ) = 
    # plt.plot( x-axis data coordinates , y-axis data coordinates  , format specifying the line style , text label assigned to the line )
    #'b-' : 'b' sets the color to blue and '-' creates a solid line.
    #'r-' : 'r' sets the color to red and '-' creates a solid line.
    plt.plot( chart_x , chart_y_training_error , 'r-' , label = 'training error' ) 
    plt.plot( chart_x , chart_y_test_error , 'b-' , label = 'test error' ) 
    # plt.axis( [ xmin , xmax , ymin , ymax ] ) : set limits for x-axis from xmin to xmax and y-axis from ymin to ymax.
    plt.axis( [ 0, len( chart_x ) , 0.0 , 1.0 ] )
    # plt.xlabel( ' ' ) : descriptive label for the horizontal x-axis
    plt.xlabel( 'training epochs' )
    # plt.ylabel( ' ' ) : descriptive label for the vertical y-axis.
    plt.ylabel('error')
    plt.legend() # show legend box ( symbol explaination table ) on the screen.
    # plt.show() # show complete chart on the screen.
    # save the whole figure into .png format
    plt.savefig('learning_curve.png') 
    print("learning_curve.png has done")
    
    
# Forward pass :    
def forward_pass( one_of_inputs_x ) : 
    # tell Python that the "hidden_layer_y" , "output_layer_y" I use later is the old array in global scope, 
    # not the new one I create in this local scope.    
    global hidden_layer_y 
    global output_layer_y
    
    # computing activation function for hidden layer
    for i , w in enumerate( hidden_layer_w ) :                                                                 # 25 * ( 784 + 1 )
        z = np.dot( w , one_of_inputs_x )                                                                     # ( 784 + 1 ) * ( 784 + 1 )  = 1
        hidden_layer_y[ i ] = np.tanh( z )  # outputs of hidden layer = activation functions for hidden layer # 25 * 1
        
    # computing activation function for output layer
    # np.concatenate( ( array0 , array1 , array2 , ...... , arrayn ) ) : join array0 , array1 , array2 , ...... , arrayn together into a new array.
    # np.concatenate( ( np.array( [ 0 , 1 ] ) , np.array( [ 2 , 3 , 4 ]  ) ) = np.array( [ 0 , 1 , 2 , 3 , 4 ] )
    inputs_of_output_perceptron = np.concatenate( ( np.array( [ 1.0 ] ) , hidden_layer_y ) )  # 1 + 25
    for i , w in enumerate( output_layer_w ) :                                                                 # 10 * ( 25 + 1 )
        z = np.dot( w , inputs_of_output_perceptron )                                                         # ( 25 + 1 ) * ( 25 + 1 ) = 1
        output_layer_y[ i ] = 1.0 / ( 1.0 + np.exp( -z ) )                                                          # 10 * 1 

# Backward pass :    
def backward_pass( one_of_outputs_y ) :
    # tell Python that the "hidden_layer_error" , "output_layer_error" I use later is the old array in global scope, 
    # not the new one I create in this local scope.  
    global hidden_layer_error 
    global output_layer_error
    
    # error of output perceptron
    for i , y in enumerate( output_layer_y ) : 
        overall_error = -2 * ( one_of_outputs_y[ i ] - y )
        output_layer_error[ i ] =  overall_error * ( y * ( 1.0 - y ) ) 
        
    # error of hidden perceptron 
    for i , y in enumerate( hidden_layer_y ) :
        # create an array of weights connecting hidden neuron [ i ] and output neurons [ 0 , 1 , ...... 9 ].
        error_weights = [] 
        for w in output_layer_w :     # 10 * ( 25 + 1 )
            # hidden neuron [ i ] will connect to the weight[ i + 1 ] of output neurons [ 0 , 1 , ...... 9 ] 
            # due to bias input must connect to bias weight ( weight[ 0 ] )
            error_weights.append( w[ i + 1 ] )
        # transfer array "error_weights" into array "error_weight_array" which is NumPy format.
        error_weight_array = np.array(error_weights)

        hidden_layer_error[ i ] = np.dot( output_layer_error , error_weight_array ) * ( 1.0 - y**2 )

# Weight Adjustment :   
def adjust_weights( one_of_inputs_x ) :
    # tell Python that the "output_layer_w" , "hidden_layer_w" I use later is the old array in global scope, 
    # not the new one I create in this local scope. 
    global output_layer_w 
    global hidden_layer_w

    # adjust weights of hidden layer
    for i , error in enumerate( hidden_layer_error ) :
        hidden_layer_w[ i ] = hidden_layer_w[ i ] + learning_rate * -( error * one_of_inputs_x ) 
    
    # adjust weights of output layer                          
    inputs_of_output_perceptron = np.concatenate( ( np.array( [ 1.0 ] ) , hidden_layer_y ) ) 
    for i , error in enumerate( output_layer_error ) :
        output_layer_w[ i ] = output_layer_w[ i ] + learning_rate * -( error * inputs_of_output_perceptron ) 



# Network training loop and test loop and show progress : 
for i in range( epoch ) : 
    
    # training loop
    # output the random sequence of index list for training examples to finish this shuffle command.
    np.random.shuffle( index_list )
    
    correct_training_results = 0 
    
    for j in index_list :
        # np.concatenate( ( array0 , array1 , array2 , ...... , arrayn ) ) : join array0 , array1 , array2 , ...... , arrayn together into a new array.
        # np.concatenate( ( np.array( [ 0 , 1 ] ) , np.array( [ 2 , 3 , 4 ]  ) ) = np.array( [ 0 , 1 , 2 , 3 , 4 ] )
        one_of_training_inputs_x = np.concatenate( ( np.array( [ 1.0 ] ) , training_inputs_x[ j ] ) )
        forward_pass( one_of_training_inputs_x ) 
        
        # arrayname.argmax() : find out the maximun value in the array and return the index of the maximun value.
        # predicted output p_y ? desired output y ( ground thruth )
        if output_layer_y.argmax() == training_outputs_y[ j ].argmax() : 
            correct_training_results += 1 
            
        backward_pass( training_outputs_y[ j ] ) 
        adjust_weights( one_of_training_inputs_x )
        
    # test loop    
    correct_test_results = 0 
    
    for j in range( len( test_inputs_x ) ) :
        # np.concatenate( ( array0 , array1 , array2 , ...... , arrayn ) ) : join array0 , array1 , array2 , ...... , arrayn together into a new array.
        # np.concatenate( ( np.array( [ 0 , 1 ] ) , np.array( [ 2 , 3 , 4 ]  ) ) = np.array( [ 0 , 1 , 2 , 3 , 4 ] )
        one_of_test_inputs_x = np.concatenate( ( np.array( [ 1.0 ] ) , test_inputs_x[ j ] ) )
        forward_pass( one_of_test_inputs_x )
        
        # arrayname.argmax() : find out the maximun value in the array and return the index of the maximun value.
        # predicted output p_y ? desired output y ( ground thruth )
        if output_layer_y.argmax() == test_outputs_y[ j ].argmax() :
            correct_test_results += 1 
            
    # show progress
    show_learning( i , 
                  correct_training_results / len( training_outputs_y ) , # accuracy of training error = correct results / total training results
                  correct_test_results / len( test_outputs_y ) )  # accuracy of test error = correct results / total test results

plot_learning() 