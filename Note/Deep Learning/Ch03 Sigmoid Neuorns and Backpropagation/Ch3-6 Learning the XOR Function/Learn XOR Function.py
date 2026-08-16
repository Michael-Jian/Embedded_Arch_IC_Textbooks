# Call NumPy library :
import numpy as np


# Prepare pair training examples : 
np.random.seed( 3 ) # set up a random seed '3' to start this shuffle command.
index_list = [ 0 , 1 , 2 , 3 ] # the inital sequence of index list for training examples.

inputs_x = [np.array( [ 1.0 , -1.0 , -1.0 ] ) , np.array( [ 1.0 , -1.0 , 1.0 ] ) , np.array( [ 1.0 , 1.0, -1.0 ] ) , np.array( [1.0 , 1.0 , 1.0 ] ) ]
outputs_y = [ 0.0 , 1.0 , 1.0 , 0.0 ] #( ground truth )

learning_rate = 0.05

# Randomly initialize input weights, predicted outputs p_y, and error terms : 
def weights_for_perceptron( num_of_input_weights ) :
    # np.zeros() : create an array / a matrix / a tensor structure with default value of each elements is 0.
    # np.zeros( x ) = create an array with default value of each elements is 0 for x elements
    # um_of_input_weights + 1 : add 1 cell for bias weight ( w0 ).
    # default vale of bias weight : weight[ i ][ 0 ] = 0
    weights = np.zeros( num_of_input_weights + 1 ) 
    
    # range( start , stop ) : creat a rule interval = [ start, stop - 1 ] , but not entity that rule.
    # range( stop ) = # range( 0 , stop ) : creat a rule interval = [ 0 , stop - 1 ] , but not entity that rule.
    # range( 1 ,... : leave index = 0 for bias weight ( w0 )
    for i in range( 1 , ( num_of_input_weights + 1  ) ) :
        # np.random.uniform( x , y ) : create a randomly and uniformly distributed value between x and y.
        # default input weights = weights[ 1 ] ~ weights[ num_of_input_weights ] = value between  -1.0 and 1.0
        weights[ i ] = np.random.uniform( -1.0, 1.0 )
    return weights
# [ perceptron N0, perceptron N1, perceptron N2 ]
weights_of_perceptrons = [ weights_for_perceptron( 2 ) , weights_for_perceptron( 2 ) , weights_for_perceptron( 2 ) ]
predicted_output_p_y_of_perceptrons = [ 0 , 0 , 0 ]
error_term_of_perceptrons = [ 0 , 0 , 0 ]


# Forward Pass : 
def forward_pass( one_of_inputs_x ) :
    # tell Python that the "predicted_output_p_y_of_perceptrons" I use later is the old array in global scope, 
    # not the new one I create in this local scope.
    global predicted_output_p_y_of_perceptrons 
    
    z0 = np.dot( weights_of_perceptrons[ 0 ], one_of_inputs_x ) # weighted sum function of perceptron N0
    predicted_output_p_y_of_perceptrons[ 0 ] = np.tanh( z0 ) # predicted output p_y of perceptron N0
    
    z1 = np.dot( weights_of_perceptrons[ 1 ], one_of_inputs_x ) # weighted sum function of perceptron N1
    predicted_output_p_y_of_perceptrons[ 1 ] = np.tanh( z1 ) # predicted output p_y of perceptron N1
    
    # inputs_of_N2 = [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ],
    # "inputs_of_N2" is a list and cannot implement matrix multiplication
    # inputs_of_N2 = np.array( [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ] )
    # "inputs_of_N2" is an array and can implement matrix multiplication 
    inputs_of_N2 = np.array( [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ] )
    z2 = np.dot( weights_of_perceptrons[ 2 ], inputs_of_N2 ) # weighted sum function of perceptron N2
    predicted_output_p_y_of_perceptrons[ 2 ] = 1.0 / ( 1.0 + np.exp( -z2 ) ) # predicted output p_y of perceptron N2


# Backward Pass :
def backward_pass( one_of_outputs_y ) :
    # tell Python that the "error_term_of_perceptrons" I use later is the old array in global scope, 
    # not the new one I create in this local scope.
    global error_term_of_perceptrons
    
    # overall_error = "loss function( MSE )" partial derivates "predicted output p_y of perceptrons N2"
    overall_error = -2 * ( one_of_outputs_y - predicted_output_p_y_of_perceptrons[ 2 ] ) 
    
    # error term of perceptronsN2 = "loss function( MSE )" partial derivates weighted sum function of perceptron N2
    error_term_of_perceptrons[ 2 ] = overall_error * ( predicted_output_p_y_of_perceptrons[ 2 ] * ( 1.0 - predicted_output_p_y_of_perceptrons[ 2 ] ) )
    
    # error term of perceptronsN1 = "loss function( MSE )" partial derivates weighted sum function of perceptron N1
    # x**y = x^y
    error_term_of_perceptrons[ 1 ] = weights_of_perceptrons[ 2 ][ 2 ] * error_term_of_perceptrons[ 2 ] * ( 1.0 - predicted_output_p_y_of_perceptrons[ 1 ]**2 ) 
    
    # x**y = x^y
    # error term of perceptronsN0 = "loss function( MSE )" partial derivates weighted sum function of perceptron N0
    error_term_of_perceptrons[ 0 ] = weights_of_perceptrons[ 2 ][ 1 ] * error_term_of_perceptrons[ 2 ] * ( 1.0 - predicted_output_p_y_of_perceptrons[ 0 ]**2 ) 


# Weights Adjustment :    
def adjust_weights( one_of_inputs_x ) :
    # tell Python that the "weights_of_perceptrons" I use later is the old array in global scope, 
    # not the new one I create in this local scope.  
    global weights_of_perceptrons
    
    weights_of_perceptrons[ 0 ] = weights_of_perceptrons[ 0 ] + learning_rate * -( error_term_of_perceptrons[ 0 ] * one_of_inputs_x )
    
    weights_of_perceptrons[ 1 ] = weights_of_perceptrons[ 1 ] + learning_rate * -( error_term_of_perceptrons[ 1 ] * one_of_inputs_x )
    
    # inputs_of_N2 = [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ],
    # "inputs_of_N2" is a list and cannot implement matrix multiplication
    # inputs_of_N2 = np.array( [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ] )
    # "inputs_of_N2" is an array and can implement matrix multiplication 
    inputs_of_N2 = np.array( [ 1.0 , predicted_output_p_y_of_perceptrons[ 0 ] , predicted_output_p_y_of_perceptrons[ 1 ] ] )
    weights_of_perceptrons[ 2 ] = weights_of_perceptrons[ 2 ] + learning_rate * -( error_term_of_perceptrons[ 2 ] * inputs_of_N2 )

# show updated weights
def show_learning() :
    print( '________________________________________________________________________________' )
    print( 'Current weights :' )
    # fruits = [ "apple", "banana", "orange" ]                            # fruits = [ "apple", "banana", "orange" ]
    # for i, j in enumerate( fruits ) :                   =               # for i in range( len ( fruits ) ) :
    #   print( "index :", i, "fruit :", j )                               #     print("index :", i, "fruit :", fruits[ i ] )
    for i, j in enumerate( weights_of_perceptrons ) :
        print( 'perceptron ' , i , ' : w0 = ' , '%5.2f' % j[ 0 ] , ' , w1 = ', '%5.2f' % j[ 1 ] , ' , w2 = ' , '%5.2f' % j[ 2 ] )
        
      

# Network training loop :
all_correct = False

# training until converged
while not all_correct :
    all_correct = True
    
    # output the random sequence of index list for training examples to finish this shuffle command.
    np.random.shuffle( index_list )
    
    # practice all training examples and update weights
    for i in index_list :
        forward_pass( inputs_x[ i ] )
        backward_pass( outputs_y[ i ] )
        adjust_weights( inputs_x[ i ] )
        show_learning() 
    
    # check if converged
    for i in range( len( inputs_x ) ) : 
        forward_pass( inputs_x[ i ] )
        print('x1 = ' , '%4.1f' %  inputs_x[ i ][ 1 ] , ' , x2 = ', '%4.1f' % inputs_x[ i ][ 2 ], ', p_y = ' , '%.4f' % predicted_output_p_y_of_perceptrons[ 2 ] )
        if( ( ( outputs_y[ i ] < 0.5 ) and ( predicted_output_p_y_of_perceptrons[ 2 ] >= 0.5 ) ) or ((outputs_y[ i ] >= 0.5 ) and ( predicted_output_p_y_of_perceptrons[ 2 ] < 0.5 ) ) ) :
            all_correct = False