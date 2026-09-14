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
model = Sequential() # 重新建立一個循序型模型 (Configuration 5)[cite: 1]

model.add(Conv2D(64, (4, 4), activation='relu', padding='same', # 增加第一層卷積，核大小 4x4，步幅預設 1[cite: 1]
                 input_shape=(32, 32, 3))) # 定義輸入影像尺寸[cite: 1]
model.add(Dropout(0.2)) # 加入 20% Dropout 以抑制過擬合[cite: 1]

model.add(Conv2D(64, (2, 2), activation='relu', padding='same', strides=(2,2))) # 增加第二層卷積，核大小 2x2，步幅加大為 2[cite: 1]
model.add(Dropout(0.2)) # 加入 20% Dropout[cite: 1]

model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) # 增加第三層卷積，通道數縮減為 32，核大小 3x3[cite: 1]
model.add(Dropout(0.2)) # 加入 20% Dropout[cite: 1]

model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) # 增加第四層卷積，通道數維持 32，核大小 3x3[cite: 1]
model.add(MaxPooling2D(pool_size=(2,2), strides=2)) # 加入最大池化層，將空間解析度減半[cite: 1]
model.add(Dropout(0.2)) # 再次加入 20% Dropout[cite: 1]

model.add(Flatten()) # 將多維特徵圖展平為 1D 陣列[cite: 1]

model.add(Dense(64, activation='relu')) # 增加隱藏的全連接層，配置 64 個神經元與 ReLU[cite: 1]
model.add(Dropout(0.2)) # 加入 20% Dropout[cite: 1]

model.add(Dense(64, activation='relu')) # 再增加一層配置 64 個神經元的全連接層[cite: 1]
model.add(Dropout(0.2)) # 加入 20% Dropout[cite: 1]

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', # 編譯模型，設定損失函數為分類交叉熵[cite: 1]
              optimizer='adam', metrics=['accuracy']) # 使用 Adam 最佳化演算法，並於訓練時監控準確率[cite: 1]

model.summary() # 輸出網路架構的摘要表與參數量統計[cite: 1]

history = model.fit( # 啟動訓練過程並記錄訓練歷史[cite: 1]
    training_images, train_labels, validation_data=(test_images, test_labels), # 輸入訓練資料與驗證資料[cite: 1]
    epochs = epoch, batch_size = batch_size , verbose=2, shuffle=True) # 依照指定的 Epochs 數量與 Batch Size 進行打亂資料的訓練[cite: 1]