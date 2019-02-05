#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as spc

'''
简单的训练mnist数据集
'''

class mnist1:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: spc.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        targets = np.array(targets_list, ndmin=2).T

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    mnist = mnist1(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("E:\Alls\软件\MNIST\mnist_train_100.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for training_data in training_data_list:
        all_values = training_data.split(",")
        inputs = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        mnist.train(inputs, targets)

    one_values = training_data.split(",")[0]
    inputs = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
    print(mnist.query(inputs))