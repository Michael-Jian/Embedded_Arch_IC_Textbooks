# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level   


# 建立 Configuration 5 的循序模型架構
model = keras.Sequential([
    # 輸入層：將 28x28 圖片展平為 784 個輸入
    keras.layers.Flatten(input_shape=(28, 28)),
    # 隱藏層：25 個神經元，使用 ReLU 激勵函數與 He normal 權重初始化
    keras.layers.Dense(25, activation='relu',
                       kernel_initializer='he_normal',
                       bias_initializer='zeros'),
    # 輸出層：10 個神經元（對應 10 個類別），使用 Softmax 與 Glorot uniform 初始化
    keras.layers.Dense(10, activation='softmax',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')
])

# 編譯模型：使用類別交叉熵 (categorical_crossentropy) 與 Adam 優化器
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 訓練模型：設定批次大小為 64 進行平行加速訓練
history = model.fit(train_images, train_labels,
                    # 提供測試集以監控泛化表現
                    validation_data=(test_images, test_labels),
                    # 設定訓練週期數
                    epochs=EPOCHS, 
                    # 設定 Mini-batch 大小為 64
                    batch_size=BATCH_SIZE,
                    # 設定詳細資訊輸出模式
                    verbose=2, 
                    # 每個 Epoch 打亂資料順序
                    shuffle=True)

