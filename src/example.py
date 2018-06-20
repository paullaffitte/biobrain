import random
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
from scipy.interpolate import spline

import biobrain

def chunk(xs, n):
    return zip(*[iter(xs)]*n)

def reducedAverage(xs, n):
    return list(map(lambda c: sum(c) / float(len(c)), chunk(xs, n)))

def plotCosts(costs, label='', points=300, smoothFactor=1):
    costsNb     = len(costs)
    limitRatio  = costsNb / (points / smoothFactor)

    if (limitRatio > 1):
        costs = reducedAverage(costs, int(limitRatio))

    x           = np.linspace(0, costsNb, len(costs))
    xNew        = np.linspace(x.min(), x.max(), points)
    costsSmooth = spline(x, costs, xNew)

    line = plt.plot(xNew, costsSmooth, label=label)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend()
    plt.ylabel('costs')
    plt.xlabel('learning iterations')

def example1():
    trainingList = [
        [.1, .1, 0.5],
        [.2, .1, 1],
        [.5, .7, 0],
        [15, -15, 1],
        [3.5, .5, 1],
        [3.5, 7.5, 0],
        [-1, -1, 0.5],
    ]

    validationData = [
        [1.1, -.9, 1],
        [.6, .42, 1],
        [-1, 5, 0],
        [.5, 2, 0],
        [5.5,  1, 1],
    ]

    costs, w1, w2, b = biobrain.train(trainingList, biobrain.sigmoid, biobrain.sigmoidD, 10000, 0.1)

    plotCosts(costs, 'old')
    plotCosts(costs, 'old10', smoothFactor=10)

    # ------------------------------------------

    trainingList = [
        ([.1, .1],      [.5]),
        ([.2, .1],      [.5]),
        ([.5, .7],      [0]),
        ([15, -15],     [1]),
        ([3.5, .5],     [1]),
        ([3.5, 7.5],    [0]),
        ([-1, -1],      [.5])
    ] * 1000;
    random.shuffle(trainingList)

    nn      = biobrain.NeuralNetwork('sigmoid')
    costs   = []

    for subTrainingList in chunk(trainingList, 100):
        nn.train(subTrainingList)
        costs.append(nn.getMeanCost(subTrainingList))

    plotCosts(costs, 'new')
    plotCosts(costs, 'new10', smoothFactor=10)

    # -------------------------------------------

    for i in range(10):
        l               = random.random() * 100 - 50
        r               = random.random() * 100 - 50
        data            = [l, r]
        expected        = 0.5 if l == r else 0 if l < r else 1

        prediction      = biobrain.makePrediction(data, biobrain.sigmoid, w1, w2, b)
        newPrediction   = nn.predict(data)

        print('expected: ' + str(expected) + ', got: ' + str('%0.2f' % prediction) + ' ' + str('%0.2f' % newPrediction) + ' ' + str('%0.2f' % (l - r)))

    plt.show()
