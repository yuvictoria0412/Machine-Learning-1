import numpy as np
import pandas as pd
import requests
import io
import math
import random

trainurl = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_train.dat"
traindata = requests.get(trainurl)
traindata.raise_for_status()
traindata = np.loadtxt(io.BytesIO(traindata.content), encoding = 'bytes')

testurl = "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_test.dat"
testdata = requests.get(testurl)
testdata.raise_for_status()
testdata = np.loadtxt(io.BytesIO(testdata.content), encoding = 'bytes')


def decision_stump(x_data, y_data):

    point_list = [float(math.floor(x_data[0]))] + [float(x) for x in x_data] + [float(math.ceil(x_data[-1]))]
    theta_list = [(point_list[i] + point_list[i + 1]) / 2 for i in range(len(point_list) - 1)]
    best_e_in = float("inf")
    best_s = 0
    best_theta = 0

    for theta in theta_list:
        for s in [-1, 1]:

            e_in = decision_stump_e_in(x_data, y_data, s, theta)
            if e_in < best_e_in:
                best_e_in = e_in
                best_s = s
                best_theta = theta
    return best_s, best_theta, best_e_in

def decision_stump_e_in(x_data, y_data, s, theta):

    e_in = 0
    for i in range(len(x_data)):
        if hypo(x_data[i], s, theta) != y_data[i]:
            e_in += 1

    return e_in / len(x_data)

def hypo(x, s, theta):
    return s * np.sign(x - theta)

def e_out_test(s, theta, dim, test):

    dimension = len(test[0]) - 1

    dat = np.array(sorted(test, key=lambda x: x[dim]))
    x_d = dat[:, dim]
    y_d = dat[:, dimension]
    e_out = decision_stump_e_in(x_d, y_d, s, theta)

    return e_out

def multiDDecision_stump(train, test):
    dimension = len(train[0]) - 1

    ret_l = []    # (s, theta, e_in)

    for i in range(dimension):
        dat = np.array(sorted(train, key=lambda x: x[i]))
        x_d = dat[:, i]
        y_d = dat[:, dimension]
        # print(x_d)
        # print(y_d)
        ret_l.append(decision_stump(x_d, y_d))

    dim = ret_l.index(min(ret_l, key=lambda r: r[2]))
    # print(dim)
    best_s, best_theta, e_in = ret_l[dim]

    e_out = e_out_test(best_s, best_theta, dim, testdata)

    return best_s, best_theta, dim, e_in, e_out

multiDDecision_stump(traindata, testdata)