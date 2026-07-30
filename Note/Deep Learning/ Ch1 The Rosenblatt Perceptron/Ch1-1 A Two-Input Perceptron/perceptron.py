# The perceptron with these specific weights implements a NAND gate

def computational_unit(w, x): # define computational_unit function.
    z = 0.0
    
    for i in range(len(w)): # compute sum of weighted inputs.
        z += x[i] * w[i]  
    if z < 0:  # compute sign function.
        return -1
    else:
        return 1

while True:
        user_input = input("Input X1 and X2 [input only -1 or 1 ] [ input 'e' for ending ]:  ")
        if user_input.lower() == 'e':
            print("Test is end ended")
            break
        
        x1_str, x2_str = user_input.split() 
        # .split()：using a blank as the boundry to divide a string into few parts 
        # "1 -1".split() will become "1", "-1"。
        x1 = float(x1_str)
        x2 = float(x2_str)
        x = [1.0, x1, x2] # X0 = 1.0 ( bias input )
        
        w = [0.9, -0.6, -0.5] # w0 = 0.9 ,w1 = -0.6 ,w2 = -0.5
        
        w0_x0 = w[0] * x[0]
        w1_x1 = w[1] * x[1]
        w2_x2 = w[2] * x[2]
        z = w0_x0 + w1_x1 + w2_x2
        y = computational_unit(w, x)
        
        x1_bool = "False" if x1 == -1 else "True" 
        x2_bool = "False" if x2 == -1 else "True" 
        y_bool  = "False" if y == -1 else "True" 
        
        print("\nResult : ")
        print(f"X1 : {x1} {x1_bool}")
        print(f"X2 : {x2} {x2_bool}")
        print(f"Y : {y} {y_bool}\n")
        