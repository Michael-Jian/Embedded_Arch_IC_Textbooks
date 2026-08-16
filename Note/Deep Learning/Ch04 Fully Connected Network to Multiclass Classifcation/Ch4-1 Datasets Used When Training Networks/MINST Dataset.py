
# 
import idx2numpy

# Insert files path : 
TRAINING_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/train-images.idx3-ubyte'
TRAINING_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/t10k-labels.idx1-ubyte'

# Read files : 
# idx2numpy.convert_from_file( filename ) : convert specific format file into NumPy 3D array format to fit for matrix multiplication.
training_images = idx2numpy.convert_from_file( TRAINING_IMAGE_FILENAME )
training_labels = idx2numpy.convert_from_file( TRAINING_LABEL_FILENAME )
test_images = idx2numpy.convert_from_file( TEST_IMAGE_FILENAME )
test_labels = idx2numpy.convert_from_file( TEST_LABEL_FILENAME )

# Print number and dimension : 
# filename.shape : introduce the dimension/tensor of the file.
# if the file is a 3 dimensions structure, then filename.shape = ( length , width , height )
print( 'number and dimension of training images : ' , training_images.shape ) # 60000 images and 28 * 28 pixel of each.
print( 'number and dimension of training labels : ' , training_labels.shape ) # 60000 one dimesion answers.
print( 'number and dimension of test images : ' , test_images.shape ) # 10000 images and 28 * 28 pixel of each.
print( 'number and dimension of test labels : ' , test_labels.shape ) # 10000 one dimesion answers.

# Print one training example : 
print( 'first training labels : ' , training_labels[ 0 ] )
print( 'first training images : ' )
# training images in train_images is not traditonal format(.png/.jpeg ) but binary format (.idx-ubyte ),
#so we use specific method to show it below.
for row in training_images[ 0 ] :
    for num in row :
        if num == 0 : 
            # print( 'x' , end = 'y' ) : after printing x , print "y" consecutively and don't change to next row.
            print( '0', end = '' ) 
        else :
            # print( 'x' , end = 'y' ) : after printing x , print "y" consecutively and don't change to next row.
            print( ' ', end = '' )
    # print( 'x' ) : after printing x , it will change to next row automatically.
    print( '' )
