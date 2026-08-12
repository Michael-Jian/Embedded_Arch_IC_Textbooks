import numpy as np

np.random.seed(3) # set up a random seed '3' to start this shuffle command.
index_list = [0, 1, 2, 3] # the inital sequence of index list for training examples.

# Define training examples.
x_train = [np.array([1.0, -1.0, -1.0]),
           np.array([1.0, -1.0, 1.0]),
           np.array([1.0, 1.0, -1.0]),
           np.array([1.0, 1.0, 1.0])]
y_train = [0.0, 1.0, 1.0, 0.0] # Output (ground truth)

 # output the random sequence of index list for training examples to finish this shuffle command.