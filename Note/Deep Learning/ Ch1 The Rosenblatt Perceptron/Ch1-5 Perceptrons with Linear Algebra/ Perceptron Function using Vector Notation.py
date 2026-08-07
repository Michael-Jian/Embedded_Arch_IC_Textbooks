import numpy as np # call NumPy library

def compute_output_vector(w, x):
    z = np.dot(w, x) # use the dot product function in NumPy library
    return np.sign(z) # use the sign activaction function in NumPy library

w = np.array([0.5, -0.3, 0.1])   # [ w0, w1, w2 ] # use the array function for weights in NumPy library
x = np.array([1.0, 3.0, 2.0])    # [ x0, x1, x2 ] # use the array function for inputs in NumPy library

y = compute_output_vector(w, x)
print(f"output: y = {y}")