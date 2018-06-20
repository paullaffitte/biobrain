import numpy as np

import utils
import activation

class NeuralNetwork:
    """NeuralNetwork"""

    defaultActivation = 'sigmoid'

    def __init__(self, activation=defaultActivation):
        self._activation = activation
        self._neuron = ([np.random.randn(), np.random.randn()], (np.random.randn()))

    def train(self, trainingList, learningRate=0.1, chunkSize=0, maxIterations=0):
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

    def predict(self, inputs):
        return self._activate(self._accumulate(inputs))

    def getMeanCost(self, trainingList):
        cost = 0
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            cost += self._calcCost(self.predict(targetInputs), targetOutputs[0])
        return cost / len(trainingList)

    def _train(self, trainingList, learningRate):
        for trainingData in trainingList:
            targetInputs, _ = trainingData

            signal = self._accumulate(targetInputs)
            prediction = self._activate(signal)

            self._learn(trainingData, signal, prediction, learningRate)

    def _learn(self, trainingData, signal, prediction, learningRate):
        targetInputs, targetOutputs = trainingData
        costD_predD     = self._calcCostD(targetOutputs[0], prediction)
        predD_signalD   = self._activate(signal, True)
        costD_zD        = costD_predD * predD_signalD

        def calibrate(value, zD_valueD):
            costD_valueD = costD_zD * zD_valueD
            return value - learningRate * costD_valueD

        def calibrateNeuron(neuron):
            weigths, biais  = self._neuron
            newWeights      = [calibrate(w, p) for w, p in zip(weigths, targetInputs)]
            biais           = calibrate(biais, 1)

            return newWeights, biais

        self._neuron = calibrateNeuron(self._neuron)

    def _calcCost(self, targetOutput, prediction):
        return np.square(prediction - targetOutput)

    def _calcCostD(self, targetOutput, prediction):
        return 2 * (prediction - targetOutput)

    def _accumulate(self, data):
        weigths, biais = self._neuron
        return sum([w * d for w, d in zip(weigths, data)]) + biais

    def _activate(self, signal, derivate=False):
        function, derivative = activation.functions.get(self._activation, self.defaultActivation)

        if (derivate):
            return derivative(signal)

        return function(signal)
