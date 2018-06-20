import random
import matplotlib.pyplot as plt
from termcolor import colored

import biobrain
import utils

def demo(brain):
    inputs = [
        [.1, .1],
        [.2, .1],
        [.5, .7],
        [15, -15],
        [3.5, .5],
        [3.5, 7.5],
        [-1, -1]
    ];

    print(colored('\nStatic examples -----', 'blue'))
    print('inputs: ' + str(inputs) + '\n' + 'predictions:' + str([brain.predict(i) for i in inputs]) + '\n')

    print(colored('\nRandom examples -----', 'blue'))
    for i in range(10):
        l               = random.random() * 100 - 50
        r               = random.random() * 100 - 50
        data            = [l, r]
        expected        = 0 if l < r else 1
        prediction      = brain.predict(data)

        print('expected: ' + str(expected) + ', got: ' + str('%0.2f' % prediction))


def loadingExample(filename):
    print(colored('LOADING EXAMPLE -----', 'green'))

    brain = biobrain.NeuralNetwork()
    brain.load(filename)
    demo(brain)

def learnigExample(filename):
    print(colored('LEARNING EXAMPLE -----', 'green'))

    trainingList = [
        ([.1, .1],      [.5]),
        ([.2, .1],      [.5]),
        ([.5, .7],      [0]),
        ([15, -15],     [1]),
        ([3.5, .5],     [1]),
        ([3.5, 7.5],    [0]),
        ([-1, -1],      [.5])
    ] * 10000;
    random.shuffle(trainingList)

    brain   = biobrain.NeuralNetwork('sigmoid')
    costs   = brain.train(trainingList, chunkSize=10, maxIterations=500)

    brain.save(filename)

    print('Last estimated mean cost: ' + str(costs.pop()))
    utils.plotCosts(costs, chunkSize=10)
    utils.plotCosts(costs, 'moving avg 5', smoothFactor=5, chunkSize=10)
    plt.show()