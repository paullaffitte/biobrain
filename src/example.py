import random
import matplotlib.pyplot as plt
from termcolor import colored

import biobrain
import utils

def calc_y(x1, x2):
    y1 = .5
    if x1 > x2:
        y1 = 1
    elif x2 > x1:
        y1 = 0
    return y1

def demo(brain):
    print(colored('\nRandom examples -----', 'blue'))
    for i in range(10):
        x1              = random.random() * 100 - 50
        x2              = random.random() * 100 - 50
        data            = [x1, x2]
        expected        = calc_y(x1, x2)
        evaluation     = brain.evaluate(data)

        print('expected: ' + str(expected) + ', got: ' + str('%0.2f' % evaluation))

def loadingExample(filename):
    print(colored('LOADING EXAMPLE -----', 'green'))

    brain = biobrain.NeuralNetwork()
    brain.load(filename)
    demo(brain)

def learnigExample(filename):
    print(colored('LEARNING EXAMPLE -----', 'green'))

    inputs = [[random.random() * 10 - 5, random.random() * 10 - 5] for x in range(10000)]
    trainingList = [([x1, x2], [calc_y(x1, x2)]) for x1, x2 in inputs]
    brain   = biobrain.NeuralNetwork('sigmoid')
    costs   = brain.train(trainingList, chunkSize=10, maxIterations=500)

    brain.save(filename)

    print('Last estimated mean cost: ' + str(costs.pop()))
    utils.plotCosts(costs, chunkSize=10)
    utils.plotCosts(costs, 'moving avg 5', smoothFactor=5, chunkSize=10)
    plt.savefig("output.png")
