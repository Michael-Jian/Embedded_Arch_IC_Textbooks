# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level   
from keras import Sequential # use the Sequential Neural Network from keras
from keras.layers import Dense # use the Dense layer from kersas
import logging  # controlling runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppressing warning logs



EPOCHS = 500 # 設定訓練週期為 500 次
BATCH_SIZE = 16 # 設定每次訓練更新的批次大小為 16

# Read and standardize the data.
boston_housing = keras.datasets.boston_housing # 載入內建的波士頓房價資料集
(raw_x_train, y_train), (raw_x_test, y_test) = boston_housing.load_data() # 拆分訓練集與測試集

x_mean = np.mean(raw_x_train, axis=0) # 針對訓練集每個特徵計算平均值
x_stddev = np.std(raw_x_train, axis=0) # 針對訓練集每個特徵計算標準差
x_train = (raw_x_train - x_mean) / x_stddev # 對訓練集資料進行標準化
x_test = (raw_x_test - x_mean) / x_stddev # 使用訓練集的平均與標準差對測試集進行標準化

# Create and train model.
model = Sequential() # 初始化空的序列模型
model.add(Dense(64, activation='relu', input_shape=[13])) # 第一隱藏層：64 個神經元，ReLU 激勵函數，接收 13 個特徵輸入
model.add(Dense(64, activation='relu')) # 第二隱藏層：64 個神經元，ReLU 激勵函數
model.add(Dense(1, activation='linear')) # 輸出層：1 個神經元，線性激勵函數（用於預測連續數值）

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error']) # 編譯模型：使用 MSE 損失函數、Adam 優化器，並印出 MAE 評估指標
model.summary() # 印出模型架構摘要

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True) # 開始訓練模型，並在測試集上進行驗證

# Print first 4 predictions.
predictions = model.predict(x_test) # 使用訓練好的模型對測試集進行預測[cite: 1]
for i in range(0, 4): # 迴圈印出前 4 筆預測結果[cite: 1]
    print('Prediction:', predictions[i], ', true value: ', y_test[i]) # 顯示預測值與實際目標值的差異[cite: 1]
    
    
    
    
    
# -------------------------------------------------------------------------------------------------- 
    # Code Snippet 6-2 How to Add L2 Regularization to the Model
from keras.regularizers import l2 # 匯入 L2 正規化模組[cite: 1]

model.add(Dense(64, activation='relu',
                kernel_regularizer=l2(0.1), # 對神經層權重 (kernel) 進行強度 0.1 的 L2 正規化[cite: 1]
                bias_regularizer=l2(0.1), # 對偏置值 (bias) 進行強度 0.1 的 L2 正規化（註：一般常見做法不針對 bias 正規化，Keras 允許拆分設定）[cite: 1]
                input_shape=[13])) # 設定 13 個特徵輸入[cite: 1]
model.add(Dense(64, activation='relu',
                kernel_regularizer=l2(0.1), # 對第二隱藏層權重進行強度 0.1 的 L2 正規化[cite: 1]
                bias_regularizer=l2(0.1))) # 對第二隱藏層偏置值進行強度 0.1 的 L2 正規化[cite: 1]
model.add(Dense(1, activation='linear',
                kernel_regularizer=l2(0.1), # 對輸出層權重進行強度 0.1 的 L2 正規化[cite: 1]
                bias_regularizer=l2(0.1))) # 對輸出層偏置值進行強度 0.1 的 L2 正規化[cite: 1]

# -------------------------------------------------------------------------------------------------- 

# Code Snippet 6-3 How to Add Dropout to the Model
from keras.layers import Dropout # 匯入 Dropout 層模組[cite: 1]


model.add(Dense(64, activation='relu', input_shape=[ 13 ])) # 第一隱藏層[cite: 1]
model.add(Dropout(0.2)) # 加入 Dropout 層，隨機關閉第一隱藏層中 20% 的神經元輸出[cite: 1]
model.add(Dense(64, activation='relu')) # 第二隱藏層[cite: 1]
model.add(Dropout(0.2)) # 加入 Dropout 層，隨機關閉第二隱藏層中 20% 的神經元輸出[cite: 1]
model.add(Dense(1, activation='linear')) # 線性輸出層[cite: 1]

# -------------------------------------------------------------------------------------------------- 