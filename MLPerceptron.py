import math
import operator
import random

from CategoricalFeature import CategoricalFeature
from ContinuousFeature import ContinuousFeature
from Observation import Observation
from Perceptron import Perceptron
from Synapse import Synapse

eta = 0.9  # tunable parameter


def classifyData(data):
    """
    @param data A data set of Observation objects
    @return classList A list of the names of the classes in the data set
    This method takes a data set and divides it up by class
    """
    classList = []
    for obs in data:
        if obs.classifier not in classList:
            classList.append(obs.classifier)

    if len(classList) > 2:
        return classList
    if len(classList) == 2:
        return [0]  # since there is only one output neuron, use 0 as the class
    else:
        raise Exception("Data must contain two or more classes. The data contain {} classes.".format(len(classList)))


def buildNet(data, hidden):
    """
    @param data The training data used to build the neural net
    @param hidden The list of hidden neurons - can be an empty list or a list of up to two values
    @return The results of trainNet - uses the network built in this method and trains it 
    This method builds a neural network composed of a Perceptron and Synapses
    """
    classes = classifyData(data)
    numFeatures = len(data[0].features)
    hidden = [int(value) for value in hidden]
    if hidden == []:
        hidden = None  # can't get a "None" argument from the command line
        layers = 1
    else:
        layers = len(hidden) + 1  # the input layer plus the number of hidden nodes is usually considered the number of layers in a neural net

    neurons = [numFeatures + 1]  # one additional neuron at the input level for the bias
    if layers > 1:
        neurons.extend(hidden)

    neurons.append(len(classes))  # output layer (1 output for 2 classes, K for >2 classes)

    network = Perceptron(neurons, layers)  # make the perceptron with the number of neurons in each layer

    synapses = []  # create the synapses for the perceptron (future steps - add the synapses to the Perceptron object)
    for index, value in enumerate(network.neurons):
        if index == len(network.neurons) - 1:  # no synapse after the output layer
            break
        synapses.append(Synapse(value, network.neurons[index + 1], None, None))  # make new synapses between the layers

    for synapse in synapses:
        synapse.setWeights()  # sets random weights between -0.01 and 0.01 for the weights, and set the delta values to 0

    return trainNet(data, classes, network, synapses)


def sigmoid(value, deriv=False):
    """
    @param value The value with which to calculate the sigmoid function
    @param deriv If the derivative of the function is to be taken
    @return The output of the sigmoid function using the given value
    This method represents a sigmoid function and it's derivative
    """
    if deriv == True:
        return (value * (1 - value))
    return (1 / (1 + math.exp(-value)))


def softmax(vector):
    """
    @param vector A list of values with which a weighted sigmoid function is generated
    @return softmax A vector of values that should be weighted toward 0 or 1, with the 1 being the max value
    This method takes a vector of dot products for each class and weights it to skew toward one maximum value
    """
    sum = 0
    for classifier, value in vector.items():
        sum += math.exp(value)

    softmax = {classifier:0 for classifier in vector}
    for classifier, value in vector.items():  # divide each e^value by the total sum of all e^values
        softmax[classifier] = math.exp(value) / sum

    return softmax


def trainNet(data, classes, network, synapses):
    """
    @param data The training data set
    @param classes List of class labels
    @param network The Perceptron object with the number of neurons and layers
    @param synapses The set of Synapse objects belonging to the Perceptron
    @return network, synapses, classes - Returns the class list and Perceptron object from the beginning as well as the updated synapses 
    This method trains the network, updating the synapse weights 
    """
#     totalError = 100
#     while totalError > 0: #got stuck in local minima? Works more effectively using 100+ cycles through the data
    for i in range(100):
        random.shuffle(data)  # shuffle the training data to go through them in a random order
        for obs in data:
#             totalError = 0
            # feed forward calculation starts here
            hidden = []  # z in the book pseudocode
            classMatrix = {classifier:0 for classifier in classes}  # r in the book pseudocode (1 if the observation belongs to the class, 0 otherwise)
            for classifier in classMatrix:
                if classifier == obs.classifier:
                    classMatrix[classifier] = 1

            input = []
            for feature in obs.features:
                input.append(feature._value)
            input.append(1)  # add a bias neuron for the input layer

            hidden.append(input)  # the first layer is the input values

            for layer in range(network.layers - 1):  # exclude the input layer and the output layer, just go through the hidden layer
                weightedSum = 0
                neurons = []
                for neuron in range(network.neurons[layer + 1] - 1):  # the number of hidden nodes including bias node, subtract 1 to add bias node
                    for index, value in enumerate(hidden[layer][:-1]):  # go through each hidden layer and calculate the weight times the calculated value plus the bias weight
                        weightedSum += (value * synapses[layer].weights[neuron][index]) + synapses[layer].weights[neuron][-1]
                    neurons.append(sigmoid(weightedSum))  # introduce nonlinearity - find the sigmoid of the weighted sum
                neurons.append(1)  # bias node
                hidden.append(neurons)

            weightedSum = {classifier:0 for classifier in classes}  # the output (classification) of the network
            for number, classifier in enumerate(classes):
                for index, neuron in enumerate(hidden[-1][:-1]):
                    weightedSum[classifier] += (neuron * synapses[-1].weights[number][index]) + synapses[-1].weights[number][-1]
            if len(classes) == 1:  # two classes output one neuron with a sigmoid value
                output = {0:(sigmoid(weightedSum[0]))}
            else:
                output = softmax(weightedSum)  # >2 classes output softmax for all of the classes
            hidden.append(output)

            # backprop starts here
            layerError = []
            for number, classifier in enumerate(classes):  # calculate the error of the second hidden layer
                error = classMatrix[classifier] - output[classifier]
                layerError.append(error)
                for index, node in enumerate(hidden[-2]):
                    synapses[-1].delta[number][index] = eta * error * node

#             totalError += sum(layerError)  # the error of the output layer

            if network.layers > 2:
                layerError = []  # supersede the outer layer's error in favor of the (more complicated) hidden layer's error
                for number, neuron in enumerate(hidden[2][:-1]):  # go through second hidden layer - calculate error of the first hidden layer
                    error = 0
                    for index, classifier in enumerate(classes):  # the derivative of the second hidden layer's error
                        error += (classMatrix[classifier] - output[classifier]) * synapses[2].weights[index][number]
                    error *= sigmoid(neuron, deriv=True)  # multiply by the derivative of the output
                    layerError.append(error)
                    for index, node in enumerate(hidden[1][:-1]):
                        synapses[1].delta[number][index] = eta * error * node

            if network.layers > 1:  # error of the hidden layer and the input
                for number, neuron in enumerate(hidden[1][:-1]):
                    error = 0
                    for index, value in enumerate(layerError):
                        error += value * synapses[1].weights[index][number]
                    error *= sigmoid(neuron, deriv=True)
                    for index, node in enumerate(hidden[0][:-1]):
                        synapses[0].delta[number][index] = eta * error * node

            for synapse in synapses:
                synapse.updateWeights()  # add the delta values to the weight values

    return(network, synapses, classes)


def predictClass(obs, network, synapses, classes):
    """
    @param obs An observation from the test data set
    @param network The neural network with the number of nodes set
    @param synapses The weight vectors (synapses) from the trained neural net
    @param classes The list of classes in the data
    This method takes an Observation from the test set and outputs the class predicted by the neural network
    """
    hidden = []  # z in the book pseudocode
    input = []
    for feature in obs.features:  # go through the network and calculate the output using the trained weights
        input.append(feature._value)

    input.append(1)
    hidden.append(input)

    for layer in range(network.layers - 1):
        weightedSum = 0
        neurons = []
        for neuron in range(network.neurons[layer + 1] - 1):  # the number of hidden nodes including bias node, subtract 1 to add bias node
            for index, value in enumerate(hidden[layer][:-1]):
                weightedSum += (value * synapses[layer].weights[neuron][index]) + synapses[layer].weights[neuron][-1]
            neurons.append(sigmoid(weightedSum))
        neurons.append(1)  # bias node
        hidden.append(neurons)

    weightedSum = {classifier:0 for classifier in classes}
    for number, classifier in enumerate(classes):
        for index, neuron in enumerate(hidden[-1][:-1]):
            weightedSum[classifier] += (neuron * synapses[-1].weights[number][index]) + synapses[-1].weights[number][-1]

    if len(classes) == 1:
        prob = sigmoid(weightedSum[0])
        if prob > 0.5:
            return 0
        else:
            return 1
    else:
        probs = softmax(weightedSum)
        return(max(probs.items(), key=operator.itemgetter(1))[0])

