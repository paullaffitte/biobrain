import numpy as np
import json

import utils
import activation

class BiobrainException(BaseException):
    """BiobrainException"""

    def __init__(self, error):
        super(BiobrainException, self).__init__()
        self.error = error


class NeuralNetwork:
    """NeuralNetwork"""

    defaultActivation = 'sigmoid'

    def __init__(self, activation=defaultActivation):
        self._activation = activation
        self._neurons = []

    def train(self, trainingList, learningRate=0.1, chunkSize=0, maxIterations=0):
        self.trainingList = trainingList

        if len(trainingList) == 0:
            raise BiobrainException('Empty training list')

        inputs, outputs = trainingList[0]
        self._setNeurons(len(inputs), len(outputs))

        if chunkSize > 0:
            trainingList = utils.chunk(trainingList, chunkSize)

        costs = []
        i = 0
        for subTrainingList in trainingList:
            self._train(subTrainingList, learningRate)
            costs.append(self.getMeanCost(subTrainingList))
            i += 1
            if maxIterations > 0 and i == maxIterations:
                break;

        return costs

    def evaluate(self, inputs):
        return [self._compute(n, inputs) for n in self._neurons]

    def getMeanCost(self, trainingList):
        costs = []
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            costs += self._calcCosts(self.evaluate(targetInputs), targetOutputs)
        return sum(costs) / len(costs)

    def save(self, filename, meanPrecision=100):
        try:
            with open(filename, 'w+') as file:
                file.write(json.dumps([self._activation, self._neurons, self.getMeanCost(self.trainingList[:meanPrecision])]))
                print('Brain saved at \'' + filename + '\'')
        except PermissionError:
            raise BiobrainException('Oops.. Permission denied!')

    def load(self, filename):
        try:
            with open(filename, 'r') as file:
                self._activation, self._neuron, meanCost = json.load(file)
                print('Brain loaded from \'' + filename + '\'\nEstimated mean cost: ' + str(meanCost))
        except FileNotFoundError:
            raise BiobrainException('Oops.. File not found!')


    def _train(self, trainingList, learningRate):
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            self._neurons = [self._learn(n, targetInputs, tOut, learningRate) for (n, tOut) in zip(self._neurons, targetOutputs)]

    def _learn(self, neuron, targetInputs, targetOutput, learningRate):
        evaluation      = self._compute(neuron, targetInputs)
        evalD_signalD   = self._compute(neuron, targetInputs, True)
        costD_evalD     = self._calcCostD(targetOutput, evaluation)
        costD_signalD   = costD_evalD * evalD_signalD

        def calibrate(value, signalD_valueD):
            costD_valueD = costD_signalD * signalD_valueD
            return value - learningRate * costD_valueD

        weigths, biais  = neuron
        newWeights      = [calibrate(w, p) for w, p in zip(weigths, targetInputs)]
        biais           = calibrate(biais, 1)

        return newWeights, biais


    def _setNeurons(self, inputs, outputs):
        self._neurons = [self._newNeuron(inputs) for _ in range(outputs)]

    def _newNeuron(self, inputs):
        return ([np.random.randn() for _ in range(inputs)], np.random.randn())

    def _calcCosts(self, targetOutputs, evaluations):
        targetEval = zip(targetOutputs, evaluations)
        return [np.square(e - t) for (t, e) in targetEval]

    def _calcCostD(self, targetOutput, evaluation):
        return 2 * (evaluation - targetOutput)

    def _compute(self, neuron, data, derivate=False):
        weigths, biais  = neuron
        signal          = sum([w * d for w, d in zip(weigths, data)]) + biais

        function, derivative = activation.functions.get(self._activation, self.defaultActivation)

        if (derivate):
            return derivative(signal)

        return function(signal)
