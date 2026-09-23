# Call Libraries :
import numpy as np # compute numerical values 
import tensorflow as tf # choose the specifc DL framework
keras = tf.keras  # use the API in high abstraction level   
from keras.applications import resnet50 # use the ResNet and relared functions.
from keras.preprocessing.image import load_img # use an image-loading function
from keras.preprocessing.image import img_to_array # use an image-transfering-array function
from keras.applications.resnet50 import decode_predictions # transfer a predicted output ( = probability ) into a message people can recognize.
import matplotlib.pyplot as plt # for showing pictures
import logging  # control runtime log messages
tf.get_logger().setLevel( logging.ERROR ) # suppress warning logs


# Load the Image and Convert Its tensor : 
# load "dog.jpg" and change the size of "dog.jpg" into 224 x 224 pixels to match the standard of ResNet50.
image = load_img( '../data/dog.jpg' , target_size = ( 224 , 224 ) ) 
# tansfer the image of  224 x 224 pixels into NumPy Array 3D format , which is 224 x 224 x 3 ( = width x height x Cout )
reshaped_image = img_to_array( image )
# X = np.expand_dims( Y , axis = k ) : 'X' = add an extra dimension with value 1 to the index k of 'Y'.
# image_np = np.expand_dims( image_np , axis = 0 ) : image_np = ( 1 x 224 x 224 x 3 ) = add 1 into the index 0 of ( 224 x 224 x 3 )
# image_np = np.expand_dims( image_np , axis = 0 ) : image_np = ( batch_size x width x height x Cout ) = add batch_size into  the index 0 of ( width x height x Cout )
np_reshaped_image = np.expand_dims( reshaped_image , axis = 0 )


# Load the Pretrained Model : 
model = resnet50.ResNet50( weights = 'imagenet' ) # put the already adjusted weights by ImageNet dataset to ResNet-50 as the pretrained model. 

# Standardize input data. (標準化輸入資料)
# image_np.copy() = the copy version of image_np，any change on the copy version of image_np won't change the original image_np.
# preprocess_input( 'X' ) : 
# change the order of output channels from RGB to BGR ,
# and each values of pixel in the ImageNet dataset need to ]
# subtract the average values of pixel ( R: 123.68, G: 116.78, B: 103.94 ) in the ImageNet dataset 
# to finish the Zero-Centering problem.
preprocess_np_reshaped_image = resnet50.preprocess_input( np_reshaped_image.copy() )


# Print Out Results :
p_y = model.predict( preprocess_np_reshaped_image ) # put the image into ResNet-50 and compute predicted outputs ( = p_y ) for each class
decoded_p_y = decode_predictions( p_y ) #  # transfer predicted outputs ( = p_y ) into messages people can recognize.
print(' predictions : ', decoded_p_y ) 
plt.imshow( np.uint8( decoded_p_y[ 0 ] ) ) # 將原始的影像陣列 (取批次中的第一張圖) 轉型為無號 8 位元整數以確保色彩顯示正確，並設定給 matplotlib
# plt.show() # show complete chart on the screen.