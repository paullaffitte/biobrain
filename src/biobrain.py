import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidD(x):
    return sigmoid(x) * (1 - sigmoid(x))

def accumulate(point, w1, w2, b):
    return point[0] * w1 + point[1] * w2 + b

def getCosts(costs, w1, w2, b, data):
    cost = 0
    for j in range(len(data)):
        p = data[j]
        p_pred = sigmoid(w1 * p[0] + w2 * p[1] + b)
        cost += np.square(p_pred - p[2])
    costs.append(cost / len(data))
    return costs

def teach(point, pred, target, signal, learningRate, activationP, w1, w2, b):
    dcost_dpred     = 2 * (pred - target)
    dpred_dsignal   = activationP(signal)

    dz_dw1  = point[0]
    dz_dw2  = point[1]
    dz_db   = 1

    dcost_dz = dcost_dpred * dpred_dsignal

    dcost_dw1   = dcost_dz * dz_dw1
    dcost_dw2   = dcost_dz * dz_dw2
    dcost_db    = dcost_dz * dz_db

    w1  = w1 - learningRate * dcost_dw1
    w2  = w2 - learningRate * dcost_dw2
    b   = b  - learningRate * dcost_db

    return w1, w2, b


def train(data, activation, activationP, iterations, learningRate):
    w1 = np.random.randn()
    w2 = np.random.randn()
    b = np.random.randn()

    costs = []

    for i in range(iterations):
        ri = np.random.randint(len(data))
        point = data[ri]

        signal = accumulate(point, w1, w2, b)
        pred = activation(signal)

        target = point[2]

        cost = np.square(pred - target)

        if i % 100 == 0:
            costs = getCosts(costs, w1, w2, b, data)

        w1, w2, b = teach(point, pred, target, signal, learningRate, activationP, w1, w2, b)

    return costs, w1, w2, b

def makePrediction(data, activation, w1, w2, b):
    return activation(accumulate(data, w1, w2, b))

class NeuralNetwork:

    defaultActivation = 'sigmoid'

    """NeuralNetwork"""
    def __init__(self, activation=defaultActivation):
        self._activation = activation
        self._neuron = ([np.random.randn(), np.random.randn()], (np.random.randn()))
        self._activationFunctions = {
            'sigmoid': (sigmoid, sigmoidD)
        };

    def train(self, trainingList, learningRate=0.1):
        for trainingData in trainingList:
            targetInputs, _ = trainingData

            signal = self._accumulate(targetInputs)
            prediction = self._activate(signal)

            self.learn(trainingData, signal, prediction, learningRate)

    def learn(self, trainingData, signal, prediction, learningRate):

# def teach(point, pred, target, signal, learningRate, activationP, w1, w2, b):
        targetInputs, targetOutputs = trainingData

        w, b = self._neuron
        w1, w2, b = teach(targetInputs, self.predict(targetInputs), targetOutputs[0], self._accumulate(targetInputs), learningRate, sigmoidD, w[0], w[1], b)
        self._neuron = ([w1, w2], b)

        # def calibrate(value, zD_valueD):
        #     costD_valueD = costD_zD * zD_valueD
        #     return value - learningRate * costD_valueD

        # def calibrateNeuron(neuron):
        #     weigths, biais = neuron
        #     zD_weigthD = []
        #     for weigth in weigths:
        #         zD_weigthD.append(calibrate(weigth, weigth))
        #     return (zD_weigthD, calibrate(biais, 1))

        # # targetInputs, targetOutputs = trainingData

        # costD_predD     = self._calcCostD(targetOutputs[0], prediction)
        # predD_signalD   = self._activate(signal, True)

        # costD_zD        = costD_predD * predD_signalD

        # self._neuron = calibrateNeuron(self._neuron)

    def predict(self, inputs):
        return self._activate(self._accumulate(inputs))

    def getMeanCost(self, trainingList):
        cost = 0
        for trainingData in trainingList:
            targetInputs, targetOutputs = trainingData
            cost += self._calcCost(self.predict(targetInputs), targetOutputs[0])
        return cost / len(trainingList)

    def _calcCost(self, targetOutput, prediction):
        return np.square(prediction - targetOutput)

    def _calcCostD(self, targetOutput, prediction):
        return 2 * (prediction - targetOutput)

    def _accumulate(self, data):
        weigths, biais = self._neuron
        return sum([w * d for w, d in zip(weigths, data)]) + biais

    def _activate(self, signal, derivate=False):
        function, derivative = self._activationFunctions.get(self._activation, self.defaultActivation)

        if (derivate):
            return derivative(signal)

        return function(signal)

