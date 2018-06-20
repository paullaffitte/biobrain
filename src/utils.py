import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
from scipy.interpolate import spline

def chunk(xs, chunkSize):
    return zip(*[iter(xs)]*chunkSize)

def reducedAverage(xs, n):
    return list(map(lambda c: sum(c) / float(len(c)), chunk(xs, n)))

def plotCosts(costs, label='mean costs over time', points=300, smoothFactor=1, chunkSize=-1):
    costsNb     = len(costs)
    limitRatio  = costsNb / (points / smoothFactor)

    if (limitRatio > 1):
        costs = reducedAverage(costs, int(limitRatio))

    x           = np.linspace(0, costsNb, len(costs))
    xNew        = np.linspace(x.min(), x.max(), points)
    costsSmooth = spline(x, costs, xNew)

    line = plt.plot(xNew, costsSmooth, label=label)
    plt.legend()
    plt.ylabel('cost')
    plt.xlabel('learning iterations' + ' (with chunk of size ' + str(chunkSize) + ')' if chunkSize != -1 else '')
