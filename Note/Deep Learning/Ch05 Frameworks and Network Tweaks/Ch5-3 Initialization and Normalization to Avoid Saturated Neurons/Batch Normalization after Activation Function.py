# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level    


# 建立一個有 64 個神經元的全連接層，並直接在此層中指定 tanh 激勵函數
keras.layers.Dense(64, activation='tanh'),
# 在激勵函數的輸出之後，加入批次正規化層
keras.layers.BatchNormalization(),