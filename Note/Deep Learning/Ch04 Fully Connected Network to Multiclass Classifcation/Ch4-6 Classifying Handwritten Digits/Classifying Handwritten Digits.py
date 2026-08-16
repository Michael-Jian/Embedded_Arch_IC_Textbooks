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
learning_rate = 0.01
epoch = 20 

# read training dataset and test dataset
def read_mnist(): 
    # idx2numpy.convert_from_file( filename ) : convert specific format file into NumPy 3D array format to fit for matrix multiplication.
    training_images = idx2numpy.convert_from_file( TRAINING_IMAGE_FILENAME )
    training_labels = idx2numpy.convert_from_file( TRAINING_LABEL_FILENAME )
    test_images = idx2numpy.convert_from_file( TEST_IMAGE_FILENAME )
    test_labels = idx2numpy.convert_from_file( TEST_LABEL_FILENAME )
        
    # reformat and standardize.
    training_inputs_x = training_images.reshape( 60000 , 784 ) # 將 60,000 張 28x28 的訓練影像重塑攤平為 1D (784 個像素)
    mean = np.mean( training_inputs_x ) # 計算訓練影像所有像素的平均值
    stddev = np.std( training_inputs_x ) # 計算訓練影像所有像素的標準差
    training_inputs_x = ( training_inputs_x - mean ) / stddev # 對訓練資料進行標準化 (減去平均除以標準差)，使數值集中在 0 附近
    
    test_inputs_x = test_images.reshape( 10000, 784 ) # 將 10,000 張 28x28 的測試影像重塑攤平為 1D
    test_inputs_x = ( test_inputs_x - mean ) / stddev # 注意！使用「訓練資料」的平均值與標準差來標準化測試資料，防止資訊外洩
    
    # One-hot encoded output.
    training_outputs_y = np.zeros( ( 60000 , 10 ) ) # 建立大小為 (60000, 10) 的全 0 陣列，準備儲存單熱編碼的訓練標籤
    test_outputs_y = np.zeros( ( 10000, 10 ) ) # 建立大小為 (10000, 10) 的全 0 陣列，準備儲存單熱編碼的測試標籤
    
    for i, y in enumerate( training_labels ) : # 走訪所有訓練標籤陣列
        training_outputs_y[ i ][ y ] = 1 # 將該筆資料對應正確數字 (y) 索引的元素設為 1，完成單熱編碼
    for i, y in enumerate( test_labels ) : # 走訪所有測試標籤陣列
        test_outputs_y[ i ][ y ] = 1 # 將對應的元素設為 1，完成單熱編碼
        
    return training_inputs_x, training_outputs_y, test_inputs_x, test_outputs_y # 回傳處理完成的輸入與標籤陣列

# Read train and test examples.
training_inputs_x, training_outputs_y, test_inputs_x, test_outputs_y = read_mnist() # 呼叫上述函式取得資料集
# the inital sequence of index list for training examples.
# 建立包含所有訓練資料索引的串列，用於後續訓練時隨機打亂順序
index_list = list( range( len( training_inputs_x ) ) )  
