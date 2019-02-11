import numpy as np
import scipy.special as spc
import matplotlib.pyplot as plt

'''
用大量的数据训练和测试mnist数据集，并用记分卡计算准确率，大致为75%
'''

class mnistTest3:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: spc.expit(x)

        self.lr = learningrate

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        target = np.array(targets_list, ndmin=2).T
        output_error = target - final_outputs
        hidden_error = np.dot(self.who.T, output_error)

        self.who+= self.lr * np.dot(output_error * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih+= self.lr * np.dot(hidden_error * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

def test(hiddennodes, learningrate, trainnum):
    input_nodes = 784
    hidden_nodes = hiddennodes
    output_nodes = 10
    learningrate = learningrate
    mnist = mnistTest3(input_nodes, hidden_nodes, output_nodes, learningrate)

    '''
    训练
    '''
    train_files = open("E:\Alls\软件\MNIST\mnist_train.csv", "r")
    train_list = train_files.readlines()
    # print("train_list size is:", len(train_list))
    train_files.close()

    for i in range(trainnum):
        for train_data in train_list:
            value_spilt = train_data.split(",")
            input_list = (np.asfarray(value_spilt[1:]) / 255.0 * 0.99) + 0.01
            target_list = np.zeros(output_nodes) + 0.01
            target_list[int(value_spilt[0])] = 0.99
            mnist.train(input_list, target_list)

    '''
    测试
    '''
    test_files = open("E:\Alls\软件\MNIST\mnist_test.csv", "r")
    test_list = test_files.readlines()
    # print("test_list size is:", len(test_list))
    test_files.close()

    scorecard = []
    for test_data in test_list:
        test_split = test_data.split(",")
        test_input_list = (np.asfarray(test_split[1:]) / 255.0 * 0.09 ) + 0.01
        final_list = mnist.query(test_input_list)
        test_zero = int(test_split[0])
        label = np.argmax(final_list)
        if label == test_zero:
            scorecard.append(1)
        else:
            scorecard.append(0)
    score_array = np.asarray(scorecard)
    score = score_array.sum() / score_array.size
    # print("记分卡为:", scorecard)
    # print("记分卡数组为:", score_array)
    # print("准确率为:", score)
    return score

if __name__ == "__main__":
    for i in range(10):
        print("第", i, "次的准确率为:", test(100, 0.3, 1))

