import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

n = 20
Ein = 0
Eout = 0
rep = 5000


def sign(x):
    if x > 0:
        return 1
    else:
        return -1
    
def generateXY():
    ''' generate data'''

    global n
    x = np.random.uniform(-1, 1, size=(n, 1))
    y = np.where(x <= 0, -1, 1)

    # generate noise
    noise_indices = np.random.choice(20, size=4, replace=False)
    y[noise_indices] = -y[noise_indices]
    
    data = np.concatenate((x, y), axis=1) # 1 column-wise concatenate
    return data.tolist()

def decision_stump(dataset):

    sort_d = sorted(dataset)    # sorted data
    min_err = len(dataset)
    best_pos = None

    for i in range(len(dataset)):
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
        return sort_d[best_pos][0], min_err
    else:
        print("no best pos")
        return 1,1

averageEin = 0
averageEout = 0

for i in range(rep):
    temp1, temp2 = decision_stump(generateXY())
    averageEin += temp2
    
# print(averageEin, averageEout)
print(averageEin := averageEin/rep/n)
averageEout = (0.5+0.3*(averageEin-1)) # Eout = 0.5 + 0.3*(Ein-1)
print(averageEout)