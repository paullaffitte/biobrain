import random
import matplotlib.pyplot as plt

import biobrain
import utils

def fakeData(n):
    inputs = [[random.random(), random.random()] for _ in range(n)]
    return [(i, calcExpected(i[0], i[1])) for i in preprocessInputs(inputs)]

def preprocessInputs(inputs):
    return [[leftIsGreater(i[0], i[1]), haveSameSign(i[0], i[1])] for i in inputs]



def haveSameSign(l, r):
    return int(l * r >= 0)

def leftIsGreater(l, r):
    return int(l > r)

def calcExpected(l, r):
    return [leftIsGreater(l, r), haveSameSign(l, r)]



def example1():
    trainingList = [
        ([.1, .1, 1],      [.5, 1]),
        ([.2, .1, 1],      [.5, 1]),
        ([.5, .7, 1],      [0, 1]),
        ([15, -15, 0],     [1, 0]),
        ([3.5, .5, 1],     [1, 1]),
        ([3.5, 7.5, 1],    [0, 1]),
        ([-1, 1, 1],       [0, 1]),
        ([2, -2, 0],       [1, 0]),
        ([-1.5, 3, 0],     [0, 0]),
        ([2, -7, 0],       [1, 0]),
    ] * 1000;

    trainingList = fakeData(10000)

    # trainingList = [
    #     ([.1, .1],      [.5, 1]),
    #     ([.2, .1],      [.5, 1]),
    #     ([.5, .7],      [0, 1]),
    #     ([15, -15],     [1, 0]),
    #     ([3.5, .5],     [1, 1]),
    #     ([3.5, 7.5],    [0, 1]),
    #     ([-1, 1],       [0, 1]),
    #     ([2, -2],       [1, 0]),
    #     ([-1.5, 3],     [0, 0]),
    #     ([2, -7],       [1, 0]),
    # ] * 1000;

    # trainingList = [
    #     ([.1, .1],      [.5]),
    #     ([.2, .1],      [.5]),
    #     ([.5, .7],      [0]),
    #     ([15, -15],     [1]),
    #     ([3.5, .5],     [1]),
    #     ([3.5, 7.5],    [0]),
    #     ([-1, 1],       [0]),
    #     ([2, -2],       [1]),
    #     ([-1.5, 3],     [0]),
    #     ([2, -7],       [1]),
    # ] * 1000;

    random.shuffle(trainingList)

    brain   = biobrain.NeuralNetwork('sigmoid')
    costs   = brain.train(trainingList, chunkSize=100)

    # -------------------------------------------

    for i in range(10):
        l               = random.random() * 100 - 50
        r               = random.random() * 100 - 50
        data            = [l, r]
        expected        = calcExpected(l, r)
        prediction      = brain.predict(data)

        print('expected: ' + str(expected) + ', got: ' + str(list(map(lambda p: '%0.2f' % p, prediction))))

    # -------------------------------------------

    print('Last mean cost: ' + str(costs.pop()))

    utils.plotCosts(costs, chunkSize=10)
    utils.plotCosts(costs, 'moving avg 5', smoothFactor=5, chunkSize=10)
    plt.show()
