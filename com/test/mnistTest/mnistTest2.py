import numpy as np
import scipy.special as spc
import matplotlib.pyplot as plt

'''
用少量的数据训练和测试mnist数据集，并用记分卡计算准确率，大致为60%
'''

class mnist2:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_functions = lambda x: spc.expit(x)

        self.lr = learningrate
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_functions(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_functions(final_inputs)

        targets = np.array(targets_list, ndmin=2).T
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))

        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_output = self.activation_functions(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_output)
        final_output = self.activation_functions(final_inputs)

        return final_output


def test(hiddensnodes, leanrningrate, trainnum):
    input_nodes = 784
    hidden_nodes = hiddensnodes
    output_nodes = 10
    learning_rate = leanrningrate
    mnist = mnist2(input_nodes, hidden_nodes, output_nodes, learning_rate)

    '''
    训练神经网络
    '''
    training_data_file = open("E:\Alls\软件\MNIST\mnist_train_100.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for i in range(trainnum):
        for training_data in training_data_list:
            all_values = training_data.split(",")
            inputs = ((np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            mnist.train(inputs, targets)

    '''
    取测试集的第一个数据测试
    '''
    querying_data_file = open("E:\Alls\软件\MNIST\mnist_test_10.csv", "r")
    querying_data_list = querying_data_file.readlines()
    querying_data_file.close()
    # test_values = querying_data_list[0].split(",")
    # image_array = (np.asfarray(test_values[1:]) / 255.0 * 0.99) + 0.01
    # print("第一个数据的标签为：", test_values[0])
    # print("第一个数据的输出值为:", mnist.query(image_array))
    #images_array = np.asfarray(test_values[1:]).reshape(28, 28)
    #plt.imshow(images_array, cmap="Greys", interpolation="None")
    #plt.show()

    '''
    用测试集所有的数据进行测试
    '''
    scorecard = []
    for query_data in querying_data_list:
        query_values = query_data.split(",")
        query_one = int(query_values[0])
        input_query = (np.asfarray(query_values[1:]) / 255.0 * 0.99) + 0.01
        final_output = mnist.query(input_query)
        label = np.argmax(final_output)
        if label == query_one:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # print("记分卡为===>", scorecard)
    scorecard_array = np.asarray(scorecard)
    # print("记分卡数组为===>", scorecard_array)
    score = scorecard_array.sum() / scorecard_array.size
    # print("测试数据集得分为===>", score)
    return score

if __name__ == "__main__":
    print(test(100, 0.3, 1))
