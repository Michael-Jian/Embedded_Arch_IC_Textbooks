# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level 


# 建立一個循序 (Sequential) 模型
model = keras.Sequential([
    # 第一層 Flatten 層
    keras.layers.Flatten(input_shape=(28, 28)),
    # 建立 25 個神經元的 Dense 層
    # 直接以字串 'glorot_uniform' 指定使用 Glorot 均勻分布來初始化權重
    keras.layers.Dense(25, activation='tanh',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros'),
    # 建立 10 個神經元的 Dense 層
    # 同樣以字串 'glorot_uniform' 指定權重初始化方式
    keras.layers.Dense(10, activation='sigmoid',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')
])