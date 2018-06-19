import biobrain

def example1():
    trainingData = [
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

    costs, w1, w2, b = biobrain.train(trainingData, biobrain.sigmoid, biobrain.sigmoidP, 10000, 0.1)

    for data in validationData:
        expected = data.pop()
        prediction = biobrain.makePrediction(data, biobrain.sigmoid, w1, w2, b)
        print ("expected: " + str(expected) + ", got: " + str("%0.2f" % prediction))
