import requests
import io
import numpy as np
from unittest import result
import matplotlib.pyplot as plt

# training data
traindata_url = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat" #input("please enter file url: ")
traindata = requests.get(traindata_url)
traindata.raise_for_status() # check http status
traindata = np.loadtxt(io.BytesIO(traindata.content), encoding = 'bytes')

# testing data
testdata_url = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat"
testdata = requests.get(testdata_url)
testdata.raise_for_status()
testdata = np.loadtxt(io.BytesIO(testdata.content), encoding = 'bytes')
# print(testdata.size)


def evaluate(best_w, w, data):
    count = error(w, data)
    
    if best_w[1] > count:
        print("ori best", best_w)
        best_w = w, count
        print("new best", best_w)
    return best_w

def error(w, data):
    count = 0
    for x in data:
        y = int(x[4])
        x = np.append([1], x[0:4])

        if int(np.sign(w.dot(x))) == 0:
            if y != -1:             # set sign(0) = -1
                count += 1
        elif int(np.sign(w.dot(x))) != y:
            count += 1
    return count

def pocket_algor(n, data, coeff):
    # Pocket Algorithm
    w = np.zeros(5)
    best_w = w, data.size
    cnt = 0

    while cnt < n:
        for x in data:
            y = int(x[4])
            x = np.append([1], x[0:4])

            if int(np.sign(w.dot(x))) == 0:
                if y != -1:
                    w += coeff * y * x
                    cnt += 1
                    best_w = evaluate(best_w, w, data)

            elif int(np.sign(w.dot(x))) != y:
                w += coeff * y * x
                cnt += 1
                best_w = evaluate(best_w, w, data)

            if cnt == n:
                break

    return best_w[0]



def PLA(n, data, coeff):
    # Perceptron Learning Algorithm
    w = np.zeros(5)
    cnt = 0
    
    while cnt < n:
        for x in data:
            y = int(x[4])
            x = np.append([1], x[0:4])

            if int(np.sign(w.dot(x))) == 0:
                if y != -1:
                    w += coeff * y * x
                    cnt += 1

            elif int(np.sign(w.dot(x))) != y:
                w += coeff * y * x
                cnt += 1

            if cnt == n:
                break

    return w

# Q18: pocket algorithm iteration 50 repeat 2000 #
error_rate = 0
n = 100
for i in range(n):
    w_pocket = pocket_algor(50, traindata, 1)
    # print(w_pocket)
    error_rate += (error(w_pocket, testdata)/testdata.size)
    # print(error_rate/(i+1))
    np.random.shuffle(traindata)  
error_rate = error_rate/n
print(error_rate)

# Q19: PLA algorithm iteration 50 repeat 2000 #
error_rate = 0
# n = 10
for i in range(n):
    w_pocket = PLA(50, traindata, 1)
    # print(w_pocket)
    error_rate += error(w_pocket, testdata)/testdata.size
    # print(error_rate/(i+1))
    np.random.shuffle(traindata)  
error_rate = error_rate/n
print(error_rate)

# Q20: pocket algorithm iteration 100 repeat 2000 #
# error_rate = 0
# n = 2000
# for i in range(n):
#     w_pocket = pocket_algor(100, traindata, 1)
#     error_rate += error(w_pocket, testdata)/testdata.size
#     np.random.shuffle(traindata)
# error_rate = error_rate/n
# print(error_rate)