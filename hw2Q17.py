import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

n = 20
Ein = 0
Eout = 0
rep = 5000

# data = np.random.uniform(-1, 1, size=(n, 1)) # 1-D
# print(data)

def sign(x):
    if x > 0:
        return 1
    else:
        return -1
    
def generateXY():
    global n
    x = np.random.uniform(-1, 1, size=(n, 1))
    y = np.where(x <= 0, -1, 1)
    # generate noise
    noise_indices = np.random.choice(20, size=4, replace=False)
    y[noise_indices] = -y[noise_indices]
    
    data = np.concatenate((x, y), axis=1) # 1 是指column-wise合併, 0 是指row-wise合併
    return data.tolist()

def decision_stump(dataset):
    # print(dataset)
    sort_d = sorted(dataset)    # sorted data
    min_err = len(dataset)
    best_pos = None

    for i in range(len(dataset)):
        # err = sum(1 for j in range(i) if sort_d[j][1] > 0) + sum(1 for j in range(i, len(dataset)) if sort_d[j][1] < 0)
        err = 0
        for j in range(i):
            if sort_d[j][1] > 0:
                err += 1
        for j in range(i, len(dataset)):
            if sort_d[j][1] < 0:
                err += 1

        if err < min_err:
            min_err = err
            best_pos = i

    if best_pos is not None:
        if best_pos < len(sort_d):
            return sort_d[best_pos][0], min_err
        else:
            return (sort_d[best_pos-1][0] + sort_d[best_pos][0]) / 2, min_err

# generateXY()
# averageEin
averageEin = 0
averageEout = 0

for i in range(rep):
    temp1, temp2 = decision_stump(generateXY())
    averageEin += temp2
    theta = temp1
    # print(theta)
    
# print(averageEin, averageEout)
print(averageEin := averageEin/rep/n)
averageEout += (0.5+0.3*(temp2-1))
print(averageEout/rep)