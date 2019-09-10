class Perceptron():

    def __init__(self, neurons, layers=2):
        """
        @param neurons A list containing the amount of neurons in each layer (input, hidden, output)
        @param layers The number of layers in the Perceptron
        Instantiates a new perceptron with a given number of layers 
        """
        self.neurons = neurons
        self.layers = layers
