import tensorflow as tf # 匯入 TensorFlow 函式庫
keras = tf.keras # 從 TensorFlow 匯入 keras 模組
import numpy as np # 匯入 numpy 用於陣列運算
import matplotlib.pyplot as plt # 匯入 matplotlib 用於繪圖與顯示影像[cite: 1]
import logging # 匯入 logging 模組[cite: 1]

tf.get_logger().setLevel(logging.ERROR) # 設定 TensorFlow 僅輸出錯誤層級的日誌[cite: 1]

cifar_dataset = keras.datasets.cifar10 # 載入 Keras 內建的 CIFAR-10 資料集[cite: 1]
(train_images, train_labels), (test_images, test_labels) = cifar_dataset.load_data() # 解包訓練集與測試集的影像與標籤[cite: 1]

print('Category:', train_labels[100]) # 印出第 100 張訓練影像的類別陣列[cite: 1]

plt.figure(figsize=(1, 1)) # 設定顯示圖片的視窗大小[cite: 1]
plt.imshow(train_images[100]) # 將第 100 張訓練影像載入繪圖物件[cite: 1]
plt.show() # 實際顯示該影像[cite: 1]