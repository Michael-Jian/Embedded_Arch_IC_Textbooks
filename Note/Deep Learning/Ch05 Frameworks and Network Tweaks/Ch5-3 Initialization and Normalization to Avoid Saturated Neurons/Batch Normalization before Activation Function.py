# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level    


# 建立一個有 64 個神經元的全連接層，此處先不指定激勵函數 (僅輸出加權總和)
keras.layers.Dense(64),
# 在加權總和後，加入批次正規化層 (BatchNormalization)
keras.layers.BatchNormalization(),
# 最後再透過 Activation 層套用 tanh 激勵函數
keras.layers.Activation('tanh'),