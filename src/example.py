import random
import matplotlib.pyplot as plt

import biobrain
import utils

def example1():
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

    brain   = biobrain.NeuralNetwork('sigmoid')
    costs   = brain.train(trainingList, chunkSize=10, maxIterations=300)

    # -------------------------------------------

    for i in range(10):
        l               = random.random() * 100 - 50
        r               = random.random() * 100 - 50
        data            = [l, r]
        expected        = 0 if l < r else 1
        prediction      = brain.predict(data)

        print('expected: ' + str(expected) + ', got: ' + str('%0.2f' % prediction))

    # -------------------------------------------

    print('Last mean cost: ' + str(costs.pop()))

    utils.plotCosts(costs, chunkSize=10)
    utils.plotCosts(costs, 'moving avg 5', smoothFactor=5, chunkSize=10)
    plt.show()
