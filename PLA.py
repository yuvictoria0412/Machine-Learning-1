## 2023/6/2 (Machine Learning Foundations)---Mathematical Foundations HW1
## Yu Hsuan Yun
## work hard, be kind, and amazing things will happen <3

from unittest import result
import numpy as np
import time
import random
import io
import requests

# training data
traindata_url = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat" #input("please enter file url: ")
train_data = requests.get(traindata_url)
train_data.raise_for_status() # check http status
train_data = np.loadtxt(io.BytesIO(train_data.content), encoding = 'bytes')
# train_data = np.loadtxt("hw1_15_train.dat.txt", dtype = float)
# print(train_data)


def PLA(data, coeff):
    # Perceptron Learning Algorithm
    count = 0
    w = np.zeros(5)
    
    while True:
        correct = 0

        for x in data:
            y = int(x[4])
            x = np.append([1], x[0:4])

            if int(np.sign(w.dot(x))) == 0:
                if y != -1:
                    w = w + coeff * y * x
                    count += 1
            elif int(np.sign(w.dot(x))) != y:
                w = w + coeff * y * x
                count += 1
            else:
                correct += 1

        if correct == len(data):
            break

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
# ans: 45

## Q16 ##

start = time.process_time()
print(PLA_fixed(train_data, 2000, 1))
end = time.process_time()
print("Q16 execute time = ", end - start)
# ans: 39.8395, 40.4425


## Q17 ##

start = time.time()
print(PLA_fixed(train_data, 2000, 0.5))
end = time.time()
print("Q17 execute time = ", end - start)
# ans: 40.0515, 39.999
