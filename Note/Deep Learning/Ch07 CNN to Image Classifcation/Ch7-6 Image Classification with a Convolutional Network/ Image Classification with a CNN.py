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

# Standardize dataset.
mean = np.mean(training_images) # 計算訓練影像所有像素的平均值[cite: 1]
stddev = np.std(training_images) # 計算訓練影像所有像素的標準差[cite: 1]

train_images = (training_images - mean) / stddev # 將訓練影像資料標準化[cite: 1]
test_images = (test_images - mean) / stddev # 使用相同的平均值與標準差將測試影像資料標準化[cite: 1]

print('mean:', mean) # 印出平均值供檢視[cite: 1]
print('stddev:', stddev) # 印出標準差供檢視[cite: 1]

# Change labels to one-hot.
train_labels = to_categorical(training_labels, num_classes=10) # 將訓練標籤轉換為 10 類別的 One-hot 編碼向量[cite: 1]
test_labels = to_categorical(test_labels, num_classes=10) # 將測試標籤轉換為 10 類別的 One-hot 編碼向量[cite: 1]