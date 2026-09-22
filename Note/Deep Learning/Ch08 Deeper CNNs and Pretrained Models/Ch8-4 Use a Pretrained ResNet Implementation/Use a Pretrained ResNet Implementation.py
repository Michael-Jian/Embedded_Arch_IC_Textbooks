# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose the specifc DL framework
keras = tf.keras  # use the API in high abstraction level   
from keras.applications import resnet50 # use the ResNet and relared functions.
from keras.preprocessing.image import load_img # use an image-loading function
from keras.preprocessing.image import img_to_array # use an image-transfering-array function
from keras.applications.resnet50 import decode_predictions # transfer the predicted output ( = probability )
import matplotlib.pyplot as plt # for showing pictures
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs