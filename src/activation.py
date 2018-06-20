import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidD(x):
    return sigmoid(x) * (1 - sigmoid(x))

functions = {
    'sigmoid': (sigmoid, sigmoidD)
};
