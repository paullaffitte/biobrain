import numpy as np

import utils
import activation

class NeuralNetwork:
    """NeuralNetwork"""

    defaultActivation = 'sigmoid'

    def __init__(self, activation=defaultActivation):
        self._activation = activation

    def train(self, trainingList, learningRate=0.1, chunkSize=0, maxIterations=0):
        if len(trainingList):
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

    def predict(self, inputs):
        return [self._compute(n, inputs)[1] for n in self._neurons]

    def getMeanCost(self, trainingList):
        cost = 0
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            cost += sum([self._calcCost(p, tOut) for p in self.predict(targetInputs) for tOut in targetOutputs]) / len(targetOutputs)
        return cost / len(trainingList)

    def _setNeurons(self, inputs, outputs):
        self._neurons = [self._addNeuron(inputs) for _ in range(outputs)]

    def _addNeuron(self, inputs):
        return ([np.random.randn() for _ in range(inputs)], np.random.randn())

    def _train(self, trainingList, learningRate):
        for trainingData in trainingList:
            targetInputs, _ = trainingData
            self._neurons = [self._learn(n, trainingData, self._compute(n, targetInputs), learningRate) for n in self._neurons]

    def _learn(self, neuron, trainingData, computation, learningRate):
        targetInputs, targetOutputs = trainingData
        signal, prediction          = computation

        costD_predD     = self._calcCostD(targetOutputs[0], prediction)
        predD_signalD   = self._activate(signal, True)
        costD_zD        = costD_predD * predD_signalD

        def calibrate(value, zD_valueD):
            costD_valueD = costD_zD * zD_valueD
            return value - learningRate * costD_valueD

        def calibrateNeuron():
            weigths, biais  = neuron
            newWeights      = [calibrate(w, p) for w, p in zip(weigths, targetInputs)]
            biais           = calibrate(biais, 1)

            return newWeights, biais

        return calibrateNeuron()

    def _calcCost(self, targetOutput, prediction):
        return np.square(prediction - targetOutput)

    def _calcCostD(self, targetOutput, prediction):
        return 2 * (prediction - targetOutput)

    def _accumulate(self, neuron, inputs):
        weigths, biais = neuron
        return sum([w * d for w, d in zip(weigths, inputs)]) + biais

    def _activate(self, signal, derivate=False):
        function, derivative = activation.functions.get(self._activation, self.defaultActivation)

        if (derivate):
            return derivative(signal)

        return function(signal)

    def _compute(self, neuron, inputs):
        signal = self._accumulate(neuron, inputs)
        prediction = self._activate(signal)
        return signal, prediction
