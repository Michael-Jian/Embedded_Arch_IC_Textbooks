# Call NumPy , matplotlib , idx2numpy libraries :  
import numpy as np # for matrix multtplication.
import matplotlib.pyplot as plt # for protraiting learning curve chart
import idx2numpy # for MNIST dataset


# Insert files path : 
TRAINING_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Appendix F Matrix of Digit Classification Network/mnist dataset/train-images.idx3-ubyte'
TRAINING_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Appendix F Matrix of Digit Classification Network/mnist dataset/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Appendix F Matrix of Digit Classification Network/mnist dataset/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Appendix F Matrix of Digit Classification Network/mnist dataset/t10k-labels.idx1-ubyte'


# Prepare pair training examples : 
np.random.seed( 7 )  # set up a random seed '7' to start this shuffle command.
learning_rate = 0.05
batch_size = 32 
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
    test_inputs_x = test_images.reshape( 10000 , 784 )
    
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
    test_outputs_y = np.zeros( ( 10000 , 10 ) )
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
# 60000 / 32 = 1875
# change index list of training input x from [ 0 , 1 , 2......59999 ] to [ 0 , 1 , 2......1874 ]
index_list = list( range( int( len( training_inputs_x ) / batch_size ) ) )


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
hidden_layer_w = weight_per_layer( 25 , 784 ) # matrix( 25 x 785 )
# the hidden_layer has 25 outputs for 25 hidden perceptrons corresponding to batch size of examples in one time.
hidden_layer_y = np.zeros( ( 25 , batch_size ) )  # matrix( 25 x batch_size )
# the hidden_layer has 25 error for 25 hidden perceptrons corresponding to batch size of examples in one time
hidden_layer_error = np.zeros( ( 25 , batch_size ) )  # matrix( 25 x batch_size )
# the output_layer has 10 output perceptrons and 25 input weights + 1 bias weight per perceptron
output_layer_w = weight_per_layer( 10, 25 ) # matrix( 10 x 26 )
# the output_layer has 10 outputs for 10 output perceptrons correspoinding to batch size of examples in one time.
output_layer_y = np.zeros( ( 10 , batch_size ) )  # matrix( 10 x batch_size )
# the output_layer has 10 error for 10 output perceptrons corresponding to batch size of examples in one time.
output_layer_error = np.zeros( ( 10, batch_size ) )  # matrix( 10 x batch_size )


# Report progress on the learning process : 
chart_x = [] # an array for X-axis data（ number of epoch ）
chart_y_training_error = []  # an array for Y-axis data（ training error ）
chart_y_test_error = []  # an array for Y-axis data（ test error ）

def show_learning( epoch, acc_of_training_error, acc_of_test_error ) :
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
    #'g-' : 'g' sets the color to green and '-' creates a solid line.
    #'y-' : 'y' sets the color to yellow and '-' creates a solid line.
    plt.plot( chart_x , chart_y_training_error , 'g-' , label = 'training error' ) 
    plt.plot( chart_x , chart_y_test_error , 'y-' , label = 'test error' ) 
    # plt.axis( [ xmin , xmax , ymin , ymax ] ) : set limits for x-axis from xmin to xmax and y-axis from ymin to ymax.
    plt.axis( [ 0, len( chart_x ) , 0.0 , 1.0 ] )
    # plt.xlabel( ' ' ) : descriptive label for the horizontal x-axis
    plt.xlabel( 'training epochs' )
    # plt.ylabel( ' ' ) : descriptive label for the vertical y-axis.
    plt.ylabel('error')
    plt.legend() # show legend box ( symbol explaination table ) on the screen.
    # plt.show() # show complete chart on the screen.
    # save the whole figure into .png format
    plt.savefig('learning curve for Mini Batch Implementation.png') 
    print("learning curve for Mini Batch Implementation.png has done")
    

#--------------------------------------------------------------------------------------------------#  
# Forward pass :        
def forward_pass( batch_of_inputs_x ) :  # matrix( 32 x 785 )
    # tell Python that the "hidden_layer_y" , "output_layer_y" I use later is the old array in global scope, 
    # not the new one I create in this local scope.
    global hidden_layer_y 
    global output_layer_y
    
    # computing activation function for hidden layer
    # np.matmul( matrix01 , matrix02 ) : standard matrix multiplication
    # matrix01 = array( n elements ) -> matrix01 = matrix( 1 x n )
    # matrix01 = matrix( n x m ) -> matrix01 = matrix( n x m )
    # matrix02 = array( m elements ) -> matrix02 = matrix( m x 1 )
    # matrix02 = matrix( n x m ) -> matrix02 = matrix( n x m )  
    # batch_of_inputs_x : matrix02 = matrix( 32 x 785 ) -> matrix02 = matrix( 32 x 785 )
    #np.transpose( batch_of_inputs_x) : matrix( 785 x 32 )
    # matrix( 25 x 785 ) * matrix( 785 x 32 ) = matrix( 25 x 32 )
    z = np.matmul( hidden_layer_w , np.transpose( batch_of_inputs_x ) ) 
    hidden_layer_y = np.tanh( z ) # matrix( 25 x 32 )
    
    # computing activation function for output layer
    # np.concatenate( ( array0 , array1 , array2 , ...... , arrayn ) ) : join array0 , array1 , array2 , ...... , arrayn together into a new array.
    # np.concatenate( ( np.array( [ 0 , 1 ] ) , np.array( [ 2 , 3 , 4 ]  ) ) = np.array( [ 0 , 1 , 2 , 3 , 4 ] )
    # np.ones() : create an array / a matrix / a tensor structure with default value of each elements is 1.
    # np.ones( x ) : create a x array structure with default value of each element is 1.
    # np.ones( x , y ) = create a  x * y matrix with default value of each element is 1.
    # np.ones( ( 1, batch_size ) create a 1 * batch size matrix with default value of each element is 1.
    # inputs_of_output_perceptron : matrix( 26 x 32 )
    inputs_of_output_perceptron = np.concatenate( ( np.ones( ( 1 , batch_size ) ) , hidden_layer_y ) ) 
    # matrix( 10 x 26 ) * matrix( 26 x 32 ) = matrix( 10 x 32 )
    z = np.matmul( output_layer_w , inputs_of_output_perceptron ) 
    output_layer_y = 1.0 / ( 1.0 + np.exp( -z ) ) # matrix( 10 x 32 )

# Backward pass :    
def backward_pass( batch_of_outputs_y ) :   # matrix( 32 x 10 )
    # tell Python that the "hidden_layer_error" , "output_layer_error" I use later is the old array in global scope, 
    # not the new one I create in this local scope. 
    global hidden_layer_error 
    global output_layer_error
    
    # error of output perceptron
    overall_error = -( np.transpose( batch_of_outputs_y ) - output_layer_y ) # matrix( 10 x 32 )
    output_layer_error = overall_error * ( output_layer_y * ( 1.0 - output_layer_y ) ) # matrix( 10 x 32 )
    
    # error of hidden perceptron 
    # np.matrix.transpose( matrix ) : transfer matrix( n x m ) into transposed matrix( m x n )   
    # matrix[ A ] = ( [A]th ) row 
    # matrix[ A  ,  B ] = ( [A]th row ^ [B]th col ) element
    # matrix[ A :  ,  B : ] = ( [A]th row ~ [last]th row ^ [B]th col ~ [last]th col ) matrix
    # matrix[: A ,  : B ] = ( [0]th row ~ [A-1]th row ^ [0]th col ~ [B-1]th col ) matrix
    # matrix[A : C , B : D ] = ( [A]th row ~ [C-1]th row ^ [B]th col ~ [D-1]th col ) matrix
    # np.matrix.transpose( output_layer_w[ : , 1 : ] ) :  matrix( 25 x 10 )
    # output_layer_error : matrix( 10 x 32 )
    # matrix( 25 x 10 ) * matrix( 10 x 32 ) = matrix( 25 x 32 )
    hidden_layer_error = ( np.matmul( np.matrix.transpose( output_layer_w[ : , 1 : ] ) , output_layer_error ) ) * ( 1.0 - hidden_layer_y**2 ) 

# Weight Adjustment :
def adjust_weights( batch_of_inputs_x ) :   # matrix( 32 x 785 )
    global output_layer_w 
    global hidden_layer_w

    # 建立一個與隱藏層權重同尺寸的暫存全零矩陣，準備用來累加「這批次中每一筆資料」算出來的梯度差值
    delta_matrix = np.zeros( ( len( hidden_layer_error[:, 0] ), len( batch_of_inputs_x[:, 0] ) ) ) 
    
    for i in range( batch_size ) : 
        # 利用迴圈走訪這 32 筆資料，計算每一筆外積並累加到 delta_matrix 中
        delta_matrix += np.outer( hidden_layer_error[:, i], batch_of_inputs_x[:, i] ) * learning_rate 
        
    delta_matrix /= batch_size # 將累加結果除以批次大小，求出這批資料的「平均梯度」
    hidden_layer_w -= delta_matrix # 用平均梯度來更新隱藏層權重
    
    # 重新建立包含整排偏誤 (1.0) 的輸出層輸入矩陣
    inputs_of_output_perceptron = np.concatenate( ( np.ones( ( 1, batch_size ) ), hidden_layer_y ) ) 
    
    # 建立暫存輸出層更新量的空矩陣
    delta_matrix = np.zeros( ( len( output_layer_error[:, 0] ), len( inputs_of_output_perceptron[:, 0] ) ) ) 
    
    for i in range( batch_size ) :
        # 計算輸出層的批次平均梯度累加
        delta_matrix += np.outer( output_layer_error[:, i], inputs_of_output_perceptron[:, i] ) * learning_rate
        
    delta_matrix /= batch_size # 取平均
    output_layer_w -= delta_matrix # 用平均梯度來更新輸出層權重
#--------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------#
# Network training loop and test loop and show progress : 
for i in range( epoch ) : 
    
    # training loop
    np.random.shuffle( index_list ) # 每個 epoch 隨機打亂「批次區塊」的處理順序
    
    correct_training_results = 0 
    
    for j in index_list :
        
        # 建立要送入網路的空白批次矩陣：輸入為 785 x BATCH_SIZE (預設值為1.0，直接滿足偏誤需求)
        batch_of_inputs_x = np.ones( ( batch_size , 785 ) )  # matrix( 32 x 785)
        # 建立真實解答的空白批次矩陣：輸出為 10 x BATCH_SIZE
        batch_of_outputs_y = np.zeros( ( batch_size , 10 ) )  # matrix( 32 x 10 )
        
        for i in range( batch_size ) :
            # 內層迴圈：從原始資料庫將這 BATCH_SIZE 筆資料逐一填入剛剛建立好的空白矩陣中
            batch_of_inputs_x[ 1 : , i ] = training_inputs_x[ : , j*32 + i ] # 60000 x 784 -> 32 x784
            batch_of_outputs_y[ : , i ] = training_outputs_y[ : , j*32 + i ] # 60000 x 10 -> 32 x 10 
            
        forward_pass( batch_of_inputs_x ) # 對「整個批次矩陣」執行一次前向傳播
        
        for k in range( batch_size ) :
            # 檢驗這 32 筆資料的預測結果(取機率最大的索引)是否與真實標籤相符
            if output_layer_y[ :, k ].argmax() == batch_of_outputs_y[ :, k ].argmax() : 
                correct_training_results += 1 
                
        backward_pass( batch_of_outputs_y ) # 針對整批資料計算誤差
        adjust_weights( batch_of_inputs_x ) # 利用剛剛算出的批次誤差來調整平均權重
        
    # test loop    
    correct_test_results = 0 
    
    # 設定 step 為 BATCH_SIZE 進行走訪。為求程式簡潔，直接捨棄最後無法整除的尾端零星測試資料
    for j in range( 0, ( len( test_inputs_x ) - batch_size ), batch_size ) :
        
        # 建立測試用的空白批次矩陣
        batch_of_inputs_x = np.ones( ( 785, batch_size ) )
        batch_of_outputs_y = np.zeros( ( 10, batch_size ) )
        
        for k in range( batch_size ) :
            # 拷貝測試資料填入批次矩陣
            batch_of_inputs_x[ 1:, k ] = test_inputs_x[ j + k ]
            batch_of_outputs_y[ :, k ] = test_outputs_y[ j + k ]
            
        forward_pass( batch_of_inputs_x ) # 前向傳播
        
        for k in range( batch_size ) :
            # 統計測試正確數
            if output_layer_y[ :, k ].argmax() == batch_of_outputs_y[ :, k ].argmax() :
                correct_test_results += 1 
                
    # show progress (此函式呼叫與原本完全相同)
    show_learning( i , 
                  correct_training_results / len( training_inputs_x ) , 
                  correct_test_results / len( test_inputs_x ) )  

plot_learning()
#--------------------------------------------------------------------------------------------------#