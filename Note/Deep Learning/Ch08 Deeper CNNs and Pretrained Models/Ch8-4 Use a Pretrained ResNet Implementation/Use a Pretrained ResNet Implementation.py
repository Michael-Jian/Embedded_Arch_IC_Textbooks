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


# Load the Image and Convert Its Tensor : 
# load "dog.jpg" and change the size of "dog.jpg" into 224 x 224 pixels to match the standard of ResNet50.
image = load_img( '../Data/dog.jpg' , target_size = ( 224 , 224 ) ) 
# tansfer the image of  224 x 224 pixels into NumPy Array 3D format , which is 224 x 224 x 3 ( = width x height x Cout )
reshaped_image = img_to_array( image )
# X = np.expand_dims( Y , axis = k ) : 'X' = add an extra dimension with value 1 to the index k of 'Y'.
# image_np = np.expand_dims( image_np , axis = 0 ) : image_np = ( 1 x 224 x 224 x 3 ) = add 1 into the index 0 of ( 224 x 224 x 3 )
# image_np = np.expand_dims( image_np , axis = 0 ) : image_np = ( batch_size x width x height x Cout ) = add batch_size into  the index 0 of ( width x height x Cout )
np_reshaped_image = np.expand_dims( reshaped_image , axis = 0 )


# Load the Pretrained Model : 
model = resnet50.ResNet50( weights = 'imagenet' ) # put the already adjusted weights by ImageNet dataset to ResNet-50 as the pretrained model. 
# image_np.copy() = the copy version of image_np，any change on the copy version of image_np won't change the original image_np.
# preprocess_input( 'X' ) : 
# change the order of output channels from RGB to BGR ,
# and each values of pixel in the ImageNet dataset need to ]
# subtract the average values of pixel ( R : 123.68 , G : 116.78 , B : 103.94 ) in the ImageNet dataset 
# to finish the Zero-Centering problem.
preprocess_np_reshaped_image = resnet50.preprocess_input( np_reshaped_image.copy() )


# Predict the Image : 
p_y = model.predict( preprocess_np_reshaped_image ) # put the image into ResNet-50 and compute predicted outputs ( = p_y ) for each class


# Print Out Results :
decoded_p_y = decode_predictions( p_y ) #  # transfer predicted outputs ( = p_y ) into messages people can recognize (  = the top 5 possible results info )

for rank , ( class_id , class_name , class_probability ) in enumerate( decoded_p_y[ 0 ] , start = 1 ) :
    print( f"Top { rank } : { class_id } ( { class_name :<20} ) -> { class_probability * 100 :.2f}% " )
# decoded_p_y[ 0 ] = the 1st bach of all decoded predicted o utputs.
# plt.imshow( X ) : matplotlib's visualized function responsible for rendering a 2D or 3D pixel matrix as a color image onto the canvas.
# plt.imshow( X )  operates under the following rules : 
# 1. in 3D color images ( Width x Height x Cout ) / 3D matrices : an floating point X needs to be in the range [ 0.0 , 1.0 ]   
#                                                                 an integer X needs to be unsigned and 8-bit ( = uint8 ) 
# 2. in 2D gray images ( Width x Height ) / 2D matrices : an integer X can be any types of integer ( Ex : uint8 , int8 , int16 , int32 , int64 ....)
# in Python, pixel matrices loaded and converted via image-processing utilities often default to the float32 data type.
# therefore , we retain values in the 0.0 to 255.0 range in this image ,which doesn't match the rule.
# np.uint8( X ) : transfer the X = 0.0 to 255.0 range into X = 0 to 255 range to match the rule.
plt.imshow( np.uint8( np_reshaped_image[ 0 ] ) ) 
# plt.show() # show complete chart on the screen.
# save the whole figure into .png format
plt.savefig('The Reshaped Image for The Dog Image.png') 
print("The Reshaped Image for The Dog Image.png has done")