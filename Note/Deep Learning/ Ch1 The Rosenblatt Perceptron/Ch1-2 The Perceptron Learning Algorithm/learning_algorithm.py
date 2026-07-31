# Perceptron structure :
def computational_unit( w, x ) :
    z = 0.0
    for i in range( len( w ) ) :
        z += x[ i ] * w[ i ] 
    if z < 0 : 
        return -1
    else:
        return 1


# '...' is equal to  "..." in Python.
# print('w0 =', 0.2 ), will output : w0 = 0.2 (there will be a space between w0 and 0.2)
# '%5.2f' % w[0] = 'printf-style string formatting' % variable
# '%5.2f' % 0.2 = 'printf-style string formatting' % number
# % ：default ahead sysmbol of printf-style string formatting in Python.
# 5：reserve 5 characters for the variable/number. if character of variable/number is not equal to 5, 
# Python will automatically add space ahead of the variable/number to make it 5 characters.
# .2 : reserved 2 decimal places after the decimal points.
# f : variable/number is floating point.
# print('w0 =', '%5.2f' % 0.2 ), will output : w0 =  0.20 (there will be 2 space between w0 and 0.2)


# Initialization code :
import random
def show_learning( w ) :
    print('w0 =', '%5.2f' % w[ 0 ], ', w1 =', '%5.2f' % w[ 1 ], ', w2 =', '%5.2f' % w[ 2 ] )

# define learning rate
learning_rate = 0.1

random.seed(7) #  '7' is defined as a specific sequence of index list made by random.shuffle().
index_list = [0, 1, 2, 3] # the specific sequence of index list for training examples.

# define training examples.
# in the index list :
# [ 0 ] : (1.0, 1.0, -1.0) -> 1.0
# [ 1 ] : (1.0, 1.0, 1.0) -> 1.0
# [ 2 ] : (1.0, -1.0, -1.0) -> 1.0
# [ 3 ] : (1.0, -1.0, 1.0) -> -1.0 
x_train = [(1.0, 1.0, -1.0), (1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (1.0, -1.0, 1.0) ] 
y_train = [1.0, 1.0, 1.0, -1.0] #  = ground truth (real output)

# define inital weights.
w = [ 0.2, -0.6, 0.25 ] # Initialize to some "random" numbers

show_learning(w)


# Training loop :
all_correct = False
while not all_correct :
    all_correct = True
    random.shuffle(index_list) # output the specific sequence of index list for training examples.
    for i in index_list :
        x = x_train[ i ] 
        y = y_train[ i ] # = ground truth (real output)
        p_y = computational_unit( w, x ) # = predicted output
        if p_y != y : # adjust weights when predicted output != real output
            for j in range( len( w ) ) :
                # if "( p_y <0 ) != ( y = +1 ) " = " ( p_y = -1 ) != ( y = +1 ) "
                # ,then " add each wj by ηxj " = " w[j] += ( y * η * x[j] ), y = +1"
                # if " ( p_y >0 ) != ( y = -1 )" = " ( p_y = +1 ) != ( y = -1 ) "
                # ,then " subtract each wj by ηxj " = " w[j] += ( y * η * x[j] ), y = -1"           
                w[ j ] += (y * learning_rate * x[j])
            all_correct = False
            show_learning(w) # show adjusted weights