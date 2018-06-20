# Biobrain

The first goal of this library was to be as easy as possible to use, thus the documentation consists of this only readme. The main feature is a machine learning algorithm based on a neural network. 

## Quickstart
First, you need to create a brain, it will hold the neural network and provide an handy api to teach your brain or tu use it.
``` python
  import biobrain

  brain = biobrain.NeuralNetwork('sigmoid') # default parameter
```
The first parameter corresponds to the activation function used by the network and is optional. 

Then you should train your brain, to do so, it's also pretty simple.
``` python
  trainingList = [
    ([.1, .1],    [.5]), # First column is inputs, second one is expected outputs
    ([.2, .1],    [.5]),
    ([.5, .7],    [0]),
    ([15, -15],   [1]),
    ([3.5, .5],   [1]),
    ([3.5, 7.5],  [0]),
    ([-1, -1],    [.5])
  ] * 10000;
  random.shuffle(trainingList) # Yeah, it's a pretty dirty way to fake some training data

  costs = brain.train(trainingList, chunkSize=10, maxIterations=500)
```

Then you can use the newly trained brain!
``` python
  data = [random.random() * 100 - 50, random.random() * 100 - 50]
  evaluation = brain.evaluate(data)
  print(evaluation) # output: [float]
```

## Persistence (save/load)
After trained your brain, you should save it to be able to load it back later.
``` python
  brain.save(filename)
  brain.load(filename)
```

## Visualisation
When you trained your brain previously, you maybe noticed that the `train` function returns some `costs`. They are the costs over the time, showing how your good brain is learning. You can display thoses costs with a shorthand function using matplotlib.

``` python
  import utils

  utils.plotCosts(costs, 'moving avg 5', smoothFactor=5, chunkSize=10)
```
![alt text](https://raw.githubusercontent.com/paullaffitte/biobrain/master/assets/visualisation.png)

## Example project
### Dependencies
To run the example project, you need to have installed `matplotlib`, `termcolor`, `numpy` and `scipy`.
### Run the example project
To see a little bit of the capacities of the library, you can run the example project ny the following command `python3 src learn filename` to start a learning session, or `python3 src load filename` to load an already trained brain.
