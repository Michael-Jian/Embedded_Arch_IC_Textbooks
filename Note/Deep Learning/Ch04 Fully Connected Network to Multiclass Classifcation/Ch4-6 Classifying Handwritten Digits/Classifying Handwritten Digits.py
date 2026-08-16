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
    # transfer 3D training_inputs_x (60000 , 28 , 28 ) into 2D training_inputs_x ( 6000 , 784 )
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
    
    return training_inputs_x , test_inputs_x , training_outputs_y , test_outputs_y 

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

# the hidden_layer has 25 hidden perceptrons and 784 input weight per perceptron
hidden_layer_w = weight_per_layer( 25 , 784 )
# the hidden_layer has 25 outputs for 25 hidden perceptrons.
hidden_layer_y = np.zeros( 25 ) 
# the hidden_layer has 25 error for 25 hidden perceptrons.

# the output_layer has 10 output perceptrons and 25 input weight per perceptron
output_layer_w = weight_per_layer( 10, 25 ) 
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
    plt.show() # show complete chart on the screen.
    
    
# Forward pass :    
def forward_pass( one_of_inputs_x ) :
    # tell Python that the "hidden_layer_y" , "output_layer_y" I use later is the old array in global scope, 
    # not the new one I create in this local scope.    
    global hidden_layer_y 
    global output_layer_y 
    
    # computing activation function for hidden layer
    for i, w in enumerate( hidden_layer_w ) :
        z = np.dot(w, one_of_inputs_x) # 利用內積運算，將輸入 x 與權重 w 相乘加總，得到加權總和 z
        hidden_layer_y[i] = np.tanh(z) # 將 z 傳入 tanh 激勵函數，並將結果儲存至隱藏層輸出陣列中
        
    hidden_output_array = np.concatenate( # 建立要傳給輸出層的輸入陣列
        (np.array([1.0]), hidden_layer_y)) # 在隱藏層所有輸出值的最前面，拼接一個常數 1.0 作為給輸出層的 bias
        
    # Activation function for output layer (輸出層的激勵函數處理)
    for i, w in enumerate(output_layer_w): # 逐一走訪輸出層每一個神經元的權重 (w)
        z = np.dot(w, hidden_output_array) # 將來自隱藏層的輸入與輸出層的權重進行內積，得到 z
        output_layer_y[i] = 1.0 / (1.0 + np.exp(-z)) # 將 z 傳入 logistic sigmoid 激勵函數，儲存為最終的預測機率

def backward_pass( one_of_outputs_y ): # 定義反向傳播函式，輸入為單熱編碼的真實解答 one_of_outputs_y
    global hidden_layer_error # 宣告使用外部變數：隱藏層的誤差項陣列
    global output_layer_error # 宣告使用外部變數：輸出層的誤差項陣列
    
    # Backpropagate error for each output neuron (計算輸出神經元的誤差)
    # and create array of all output neuron errors.
    for i, y in enumerate(output_layer_y): # 逐一走訪每個輸出神經元的預測值 y
        error_prime = -(one_of_outputs_y[i] - y) # Loss derivative: 計算損失函數的導數 (即預測值 - 真實值)
        derivative = y * (1.0 - y) # Logistic derivative: 計算 logistic sigmoid 函數的導數
        output_layer_error[i] = error_prime * derivative # 兩者相乘，得到該輸出神經元的誤差項
        
    # Backpropagate error for hidden neuron. (計算隱藏層神經元的誤差)
    for i, y in enumerate(hidden_layer_y): # 逐一走訪每個隱藏層神經元的輸出 y
        # Create array weights connecting the output of
        # hidden neuron i to neurons in the output layer.
        error_weights = [] # 建立空清單，用來收集「所有連接到這個隱藏神經元」的輸出層權重
        for w in output_layer_w: # 走訪輸出層的所有神經元權重
            error_weights.append(w[i+1]) # 取出對應的權重 (索引 i+1 是為了跳過最前面的 bias 權重)
        error_weight_array = np.array(error_weights) # 將收集到的權重轉換成 NumPy 陣列
        
        derivative = 1.0 - y**2 # tanh derivative: 計算 tanh 激勵函數的導數
        weighted_error = np.dot(error_weight_array, # 計算回傳的加權誤差總和：
                                output_layer_error) # 將剛剛收集的權重陣列與「輸出層誤差項」做內積
        hidden_layer_error[i] = weighted_error * derivative # 加權誤差總和乘上激勵函數導數，得到該隱藏神經元的誤差項

def adjust_weights(one_of_inputs_x): # 定義權重調整函式，輸入為原始的資料特徵 x
    global output_layer_w # 宣告使用外部變數：輸出層的權重
    global hidden_layer_w # 宣告使用外部變數：隱藏層的權重
    
    # 更新隱藏層權重
    for i, error in enumerate(hidden_layer_error): # 逐一走訪隱藏層的誤差項
        hidden_layer_w[i] -= (one_of_inputs_x * learning_rate # 權重扣除 (輸入特徵 * 學習率 * 誤差項) 來完成梯度下降更新
                              * error) # Update all weights
                              
    hidden_output_array = np.concatenate( # 再次重新拼接隱藏層的輸出，加上 1.0 作為 bias，當作輸出層更新權重時的「輸入」
        (np.array([1.0]), hidden_layer_y))
        
    # 更新輸出層權重
    for i, error in enumerate(output_layer_error): # 逐一走訪輸出層的誤差項
        output_layer_w[i] -= (hidden_output_array # 權重扣除 (隱藏層輸入特徵 * 學習率 * 誤差項) 來完成更新
                              * learning_rate
                              * error) # Update all weights  
    
    