## 2023/6/2 (Machine Learning Foundations)---Mathematical Foundations HW1
## Yu Hsuan Yun
## work hard, be kind, and amazing things will happen <3

from unittest import result
import numpy as np
import time
import random

train_data = np.loadtxt("hw1_15_train.dat.txt", dtype = float)
# print(train_data)

def find_mistake(w, data):
    # Standard PLA
    result = None

    for x in data:
        y = int(x[4])
        x = np.append([1], x[0:4])

        if int(np.sign(w.dot(x))) == 0 and y != -1:
            result = np.array(x[0:5]), y
            break
        if int(np.sign(w.dot(x))) != y:
            result = np.array(x[0:5]), y
            break
        
    return result

def PLA(data, coeff):
    # Perceptron Learning Algorithm
    count = 0
    w = np.zeros(5)
    
    while True:
        res = find_mistake(w, train_data)
        if res is not None:
            x, y = res
            w += coeff * y * x
            count += 1
        else:
            break

    # return w  ## w is g (weight)
    return count


def PLA_fixed(data, n, coeff):
    # PLA with fixed, random data order

    result = 0
    
    for i in range(n):
        np.random.shuffle(data)
        result += PLA(data, coeff)

    result /= n
    return result

## Q15 ##

start = time.time()
print(PLA(train_data, 1))
end = time.time()
print("Q15 execute time = ", end - start)

## Q16 ##

start = time.process_time()
print(PLA_fixed(train_data, 2000, 1))
end = time.process_time()
print("Q16 execute time = ", end - start)


## Q17 ##

start = time.time()
print(PLA_fixed(train_data, 2000, 0.5))
end = time.time()
print("Q17 execute time = ", end - start) 
