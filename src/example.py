import random
import matplotlib.pyplot as plt

import biobrain
import utils

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

    utils.plotCosts(costs, 'old')
    utils.plotCosts(costs, 'old10', smoothFactor=10)

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
    costs   = nn.train(trainingList, chunkSize=100, maxIterations=300)
    # costs   = []

    # i = 0
    # for subTrainingList in utils.chunk(trainingList, 1):
    #     nn.train(subTrainingList)
    #     costs.append(nn.getMeanCost(subTrainingList))
    #     i += 1
    #     if i == 300:
    #         break;

    utils.plotCosts(costs, 'new')
    utils.plotCosts(costs, 'new10', smoothFactor=10)

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
