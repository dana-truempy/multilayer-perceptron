import random


class Synapse():

    def __init__(self, input, output, weights, delta):
        """
        @param input The number of input nodes
        @param output The number of output nodes
        @param weights Will be empty when first declared, but is set by setWeights - the list of weights of the synapses between two layers of the net
        @param delta Also set by setWeights - to 0 at first. The weight changes for each round of the algorithm.
        Instantiates a new synapse (weight matrix) with a list of weights 
        """
        self.input = input
        self.output = output
        self.weights = weights  # v and w in the book pseudocode
        self.delta = delta  # deltav, deltaw in the book pseudocode

    def setWeights(self):
        """
        Sets small, random starting weights and 0 delta weights for the synapse
        """

        weight = [[] for value in range(self.output)]
        delta = [[] for value in range(self.output)]
        for index, list in enumerate(weight):
            weight[index] = [random.uniform(-0.01, 0.01) for index in range(self.input)]
            delta[index] = [0 for index in range(self.input)]

        self.weights = weight
        self.delta = delta

    def updateWeights(self):
        """
        Adds the delta values to the weight value
        """
        for number, list in enumerate(self.weights):
            for index, weight in enumerate(list):
                self.weights[number][index] += self.delta[number][index]
