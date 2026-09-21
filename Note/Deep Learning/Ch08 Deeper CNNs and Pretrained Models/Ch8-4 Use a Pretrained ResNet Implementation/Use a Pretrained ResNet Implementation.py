import numpy as np # 匯入 NumPy 套件，用於處理陣列與張量
from tensorflow.keras.applications import resnet50 # 從 Keras 匯入 ResNet50 模型與相關功能
from tensorflow.keras.preprocessing.image import load_img # 匯入載入影像的工具函數
from tensorflow.keras.preprocessing.image import img_to_array # 匯入將影像轉換為陣列的工具函數
from tensorflow.keras.applications.resnet50 import \
decode_predictions # 匯入將預測機率解碼為人類可讀文字標籤的函數
import matplotlib.pyplot as plt # 匯入 matplotlib 用於顯示圖片
import tensorflow as tf # 匯入 TensorFlow 核心套件
import logging # 匯入 logging 模組以控制日誌輸出
tf.get_logger().setLevel(logging.ERROR) # 將 TensorFlow 的日誌層級設為 ERROR，以隱藏不必要的警告訊息