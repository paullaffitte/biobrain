#!/usr/bin/env python3

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidP(x):
    return sigmoid(x) * (1 - sigmoid(x))

def accumulate(point, w1, w2, b):
    return point[0] * w1 + point[1] * w2 + b

def getCosts(costs, w1, w2, b, data):
    cost = 0
    for j in range(len(data)):
        p = data[j]
        p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
        cost += np.square(p_pred - p[2])
    costs.append(cost)
    return costs

def teach(point, pred, target, signal, learningRate, activationP, w1, w2, b):
    dcost_dpred     = 2 * (pred - target)
    dpred_dsignal   = activationP(signal)

    dz_dw1  = point[0]
    dz_dw2  = point[1]
    dz_db   = 1

    dcost_dz = dcost_dpred * dpred_dsignal

    dcost_dw1   = dcost_dz * dz_dw1
    dcost_dw2   = dcost_dz * dz_dw2
    dcost_db    = dcost_dz * dz_db

    w1  = w1 - learningRate * dcost_dw1
    w2  = w2 - learningRate * dcost_dw2
    b   = b  - learningRate * dcost_db

    return w1, w2, b


def train(data, activation, activationP, iterations, learningRate):
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    costs = []

    for i in range(iterations):
        ri = np.random.randint(len(data))
        point = data[ri]

        signal = accumulate(point, w1, w2, b)
        pred = activation(signal)

        target = point[2]

        cost = np.square(pred - target)

        if i % 100 == 0:
            costs = getCosts(costs, w1, w2, b, data)

        w1, w2, b = teach(point, pred, target, signal, learningRate, activationP, w1, w2, b)

    return costs, w1, w2, b

def makePrediction(data, activation, w1, w2, b):
    return activation(accumulate(data, w1, w2, b))