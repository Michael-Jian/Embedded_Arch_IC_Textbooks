# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose the specifc DL framework
keras = tf.keras  # use the API in high abstraction level   
from keras.utils import to_categorical # use a one hot coding utensil
from keras.models import Sequential # use a Sequential Neural Network from keras
from keras.layers import Conv2D # use a convolutional layer from kersas
from keras.layers import Flatten # use a 1D flat layer from kersas
from keras.layers import Dense # use a fully connected layer from kersas
from keras.layers import Dropout # use Iverted Drpout from keras
from keras.layers import MaxPooling2D # use Max Pooling from keras
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

# standardize images : 
mean = np.mean( training_images ) 
stddev = np.std( training_images ) 
training_images = ( training_images - mean ) / stddev
test_images = ( test_images - mean ) / stddev
print( ' mean : ' , mean ) 
print('stdandard dieviation : ' , stddev ) 

# transfer lables into one hot code : 
# to_categorical( the original data , clsses of the original data ) 
train_labels = to_categorical( training_labels , num_classes = 10 )
test_labels = to_categorical( test_labels , num_classes = 10 ) 



# Model with two convolutional and one fully connected layer.
model = Sequential() # 建立一個空的循序型神經網路模型[cite: 1]

model.add(Conv2D(64, (5, 5), strides=(2,2), # 加入第一層 2D 卷積層，配置 64 個通道，卷積核大小為 5x5，步幅為 2x2[cite: 1]
                 activation='relu', padding='same', # 使用 ReLU 激發函數，設定 padding='same' 使填補後能對齊邊界像素[cite: 1]
                 input_shape=(32, 32, 3), # 宣告輸入影像的維度為 32x32 像素與 3 個顏色通道[cite: 1]
                 kernel_initializer='he_normal', bias_initializer='zeros')) # 設定卷積核權重與偏差項的初始化方式[cite: 1]

model.add(Conv2D(64, (3, 3), strides=(2,2), # 加入第二層 2D 卷積層，配置 64 個通道，卷積核縮小為 3x3，步幅為 2x2[cite: 1]
                 activation='relu', padding='same', # 同樣使用 ReLU 激發函數與 'same' 邊界填補[cite: 1]
                 kernel_initializer='he_normal', bias_initializer='zeros')) # 設定權重與偏差項初始化[cite: 1]

model.add(Flatten()) # 加入展平層，將 3D 特徵圖轉換為 1D 向量以供後續全連接層使用[cite: 1]

model.add(Dense(10, activation='softmax', # 加入最終的全連接層，配置 10 個神經元對應 10 個類別，並使用 Softmax 輸出機率分佈[cite: 1]
                kernel_initializer='glorot_uniform', bias_initializer='zeros')) # 設定權重與偏差項初始化[cite: 1]

model.compile(loss='categorical_crossentropy', # 編譯模型，設定損失函數為分類交叉熵[cite: 1]
              optimizer='adam', metrics=['accuracy']) # 使用 Adam 最佳化演算法，並於訓練時監控準確率[cite: 1]

model.summary() # 輸出網路架構的摘要表與參數量統計[cite: 1]

history = model.fit( # 啟動訓練過程並記錄訓練歷史[cite: 1]
    training_images, train_labels, validation_data=(test_images, test_labels), # 輸入訓練資料與驗證資料[cite: 1]
    epochs = epoch, batch_size = batch_size , verbose=2, shuffle=True) # 依照指定的 Epochs 數量與 Batch Size 進行打亂資料的訓練[cite: 1]