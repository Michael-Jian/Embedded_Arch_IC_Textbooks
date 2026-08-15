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

# Function to read dataset.
def read_mnist(): # Read training dataset and test dataset : 
    # idx2numpy.convert_from_file( filename ) : convert specific format file into NumPy 3D array format to fit for matrix multiplication.
    training_images = idx2numpy.convert_from_file( TRAINING_IMAGE_FILENAME )
    training_labels = idx2numpy.convert_from_file( TRAINING_LABEL_FILENAME )
    test_images = idx2numpy.convert_from_file( TEST_IMAGE_FILENAME )
    test_labels = idx2numpy.convert_from_file( TEST_LABEL_FILENAME )
        
    # Reformat and standardize.
    x_train = train_images.reshape(60000, 784) # 將 60,000 張 28x28 的訓練影像重塑攤平為 1D (784 個像素)
    mean = np.mean(x_train) # 計算訓練影像所有像素的平均值
    stddev = np.std(x_train) # 計算訓練影像所有像素的標準差
    x_train = (x_train - mean) / stddev # 對訓練資料進行標準化 (減去平均除以標準差)，使數值集中在 0 附近
    
    x_test = test_images.reshape(10000, 784) # 將 10,000 張 28x28 的測試影像重塑攤平為 1D
    x_test = (x_test - mean) / stddev # 注意！使用「訓練資料」的平均值與標準差來標準化測試資料，防止資訊外洩
    
    # One-hot encoded output.
    y_train = np.zeros((60000, 10)) # 建立大小為 (60000, 10) 的全 0 陣列，準備儲存單熱編碼的訓練標籤
    y_test = np.zeros((10000, 10)) # 建立大小為 (10000, 10) 的全 0 陣列，準備儲存單熱編碼的測試標籤
    
    for i, y in enumerate(train_labels): # 走訪所有訓練標籤陣列
        y_train[i][y] = 1 # 將該筆資料對應正確數字 (y) 索引的元素設為 1，完成單熱編碼
    for i, y in enumerate(test_labels): # 走訪所有測試標籤陣列
        y_test[i][y] = 1 # 將對應的元素設為 1，完成單熱編碼
        
    return x_train, y_train, x_test, y_test # 回傳處理完成的輸入與標籤陣列

# Read train and test examples.
x_train, y_train, x_test, y_test = read_mnist() # 呼叫上述函式取得資料集
index_list = list(range(len(x_train)))  # the inital sequence of index list for training examples.
