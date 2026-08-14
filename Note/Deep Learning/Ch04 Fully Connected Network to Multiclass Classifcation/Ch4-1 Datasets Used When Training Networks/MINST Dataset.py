import idx2numpy

# Insert files path : 
TRAIN_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/train-images.idx3-ubyte'
TRAIN_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/train-labels.idx1-ubyte'
TEST_IMAGE_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/t10k-images.idx3-ubyte'
TEST_LABEL_FILENAME = '/home/michael/Embedded_Arch_IC_Textbooks/Note/Deep Learning/Ch04 Fully Connected Network to Multiclass Classifcation/Ch4-1 Datasets Used When Training Networks/mnist dataset/t10k-labels.idx1-ubyte'

# Read files : 
# idx2numpy.convert_from_file( filename ) : convert specific format file into the format fit for matrix multiplication.
train_images = idx2numpy.convert_from_file( TRAIN_IMAGE_FILENAME )
train_labels = idx2numpy.convert_from_file( TRAIN_LABEL_FILENAME )
test_images = idx2numpy.convert_from_file( TEST_IMAGE_FILENAME )
test_labels = idx2numpy.convert_from_file( TEST_LABEL_FILENAME )

# Print dimensions : 
# filename.shape : introduce the number of data and dimension of data in this file.
print( 'dimensions of training images : ' , train_images.shape ) # 6000 images and 28 * 28 pixel of each.
print( 'dimensions of training labels : ' , train_labels.shape ) # 6000 answers and no pixel of each.
print( 'dimensions of test images : ' , test_images.shape ) # 10000 images and 28 * 28 pixel of each.
print( 'dimensions of test labels : ' , test_labels.shape ) # 10000 answers and no pixel of each.

# Print one training example : 
print( 'first training labels : ' , train_labels[ 0 ] )
print( 'first training images : ' )
# training images in train_images is not traditonal format(.png/.jpeg ) but binary format (.idx-ubyte ),
#so we use specific method to show it below.
for i in train_images[ 0 ] :
    for j in i :
        if j > 0 :
            print( '*', end = '' )
        else :
            print( ' ', end = '' )
    print( '' )
