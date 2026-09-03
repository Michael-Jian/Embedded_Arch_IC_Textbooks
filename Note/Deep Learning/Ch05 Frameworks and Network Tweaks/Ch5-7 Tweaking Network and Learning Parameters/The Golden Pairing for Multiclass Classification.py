# Call libraries :
import numpy as np # numerical computing                        
import tensorflow as tf # chosing specifc DL framework                  
keras = tf.keras  # using API in high abstraction level   


# Initializing (hyper)parameters : 
epoch = 14
batch_size = 64

# Load and Prepare training dataset and test dataset :
# load MNIST dataset
mnist = keras.datasets.mnist
( train_images, train_labels ) , ( test_images , test_labels ) = mnist.load_data()



neural_network = keras.Sequential ( [
    # input layer : 
    # reshaping multi-dimensional inputs into one dimensional inputs.
    keras.layers.Flatten( input_shape = ( 28 , 28 ) ) , 
    # hidden layer : 
    # activation function : ReLU fumnction.
    # initializer of input weights : He fuuction with nrmal distribution.
    # initializer of bias weights  : zeros.
    keras.layers.Dense( 25 , activation='relu' , kernel_initializer='he_normal' , bias_initializer='zeros' ) ,
    # 輸出層：10 個神經元（對應 10 個類別），使用 Softmax 與 Glorot uniform 初始化
    keras.layers.Dense(10, activation='softmax',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros')
])


# Training neural network :


# creating a compiler with type of loss function , type of optimizer , and type of supervised metric.
neural_network.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# creating a trainer with training datasets , test datasets , epoch , batch size , type of verbosity , shuffle mechanism
neural_network_trainer = neural_network.fit(train_images, train_labels,
                    # 提供測試集以監控泛化表現
                    validation_data=(test_images, test_labels),
                    # 設定訓練週期數
                    epochs=epoch, 
                    # 設定 Mini-batch 大小為 64
                    batch_size=batch_size,
                    # 設定詳細資訊輸出模式
                    verbose=2, 
                    # 每個 Epoch 打亂資料順序
                    shuffle=True)

