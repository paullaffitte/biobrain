# biobrain

The first goal of this library was to be as easy as possible to use, thus the documentation consists of this only readme. The main feature is a machine learning algorithm based on a neural network. 

## Quickstart
First, you need to create a brain, it will hold the neural network and provide an handy api to teach your brain or tu use it.
``` python
  brain = biobrain.NeuralNetwork('sigmoid') # default parameter
```
The first parameter corresponds to the activation function used by the network and is optional. 

Then you should train your brain, to do so, it's also pretty simple.
``` python
  brain.train(trainingList, chunkSize=10, maxIterations=500)
```

Then you can use the newly trained brain!
``` python
  data = [random.random() * 100 - 50, random.random() * 100 - 50]
  brain.evaluate(data)
```
