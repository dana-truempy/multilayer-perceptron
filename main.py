"""
@param sys.argv[1] The name of the file containing the data set to be divided into training/testing data
@param sys.argv[2] The name of the file to write out the model and classifications
@param sys.argv[3:] The number of hidden neurons (if given as a list of two, two hidden layers are used)
This is a main method to call all of the other scripts, takes one of the five data sets for this assignment and creates a 
multilayer perceptron for each, then calculates prediction accuracy using five-fold cross-validation
"""

from csv import writer
import random
import sys

from CrossValidation import kFoldCrossValidation
import MLPerceptron
import cancerProcess
import glassProcess
import irisProcess
import soybeanProcess
import votesProcess

if 'iris' in sys.argv[1].lower():
    dataset = irisProcess.process(sys.argv[1])

elif 'cancer' in sys.argv[1].lower():
    dataset = cancerProcess.process(sys.argv[1])

elif 'votes' in sys.argv[1].lower():
    dataset = votesProcess.process(sys.argv[1])

elif 'glass' in sys.argv[1].lower():
    dataset = glassProcess.process(sys.argv[1])

elif 'soybean' in sys.argv[1].lower():
    dataset = soybeanProcess.process(sys.argv[1])

crossFolds = kFoldCrossValidation(dataset)
train = []
accuracy = []  # get accuracy values
write = []  # write out learned model and classifications for test

random.seed()
for i in range(len(crossFolds)):
    train = []
    predicts = []  # only want to write the last crossfold's predictions

    predicts.append(["Actual class", "Predicted class"])
    for crossFold in crossFolds[:i] + crossFolds[i + 1:]:  # use all crossfolds but the test one as training set
        train.extend(crossFold)

    network, synapses, classes = MLPerceptron.buildNet(train, sys.argv[3:])

    line = []
    line.append(["Synapse weights: "])  # write the weights of the trained model
    for index, synapse in enumerate(synapses):
        line.append("Synapse {}:".format(index))
        line.append(synapse.weights)
    write.append(line)

    mistakes = 0  # now that the model has been created and recorded, test it
    for obs in crossFolds[i]:  # use other crossfold for testing
        prediction = MLPerceptron.predictClass(obs, network, synapses, classes)
        predicts.append([obs.classifier, prediction])
        if prediction != obs.classifier:  # if the network classifies incorrectly, list as mistake
            mistakes += 1
    accuracy.append((len(crossFolds[i]) - mistakes) / len(crossFolds[i]))  # get the accuracies of each cross-validation run

write.append(["Accuracy for each of five crossfolds"])  # write out accuracy of each run
write.append(accuracy)
write.append(predicts)

print("Average accuracy over five-fold cross-validation:")
print(sum(accuracy) / len(accuracy))  # print the average accuracy of the five runs

with open(sys.argv[2], 'w', newline='') as File:  # write out the file containing the model and classes
    writer = writer(File)
    writer.writerows(write)
