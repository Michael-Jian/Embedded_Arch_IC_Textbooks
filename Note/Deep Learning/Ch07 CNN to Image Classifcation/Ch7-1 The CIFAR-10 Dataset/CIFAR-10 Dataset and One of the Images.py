# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose specifc DL framework
keras = tf.kersa  # use API in high abstraction level        
import matplotlib.pyplot as plt  # for protraiting images
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs



cifar_dataset = keras.datasets.cifar10 # 將 keras 內建的 CIFAR-10 資料集指定給變數

# 載入資料，並解構分配給訓練集與測試集的影像和標籤變數
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data()

print('Category:', train_labels [100]) # 印出訓練集中第 100 號影像的分類標籤

plt.figure(figsize=(1, 1)) # 創建一個大小為 1x1 吋的畫布
plt.imshow(train_images [100]) # 繪製訓練集中第 100 號的影像內容
plt.show() # 將繪製好的影像顯示在畫面上 