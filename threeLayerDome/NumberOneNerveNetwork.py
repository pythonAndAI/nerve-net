#!/usr/bin/python3
import numpy as np
import scipy.special as spc


class neuralNetwork:
    # 初始化数据
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入层、隐藏层、输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate

        # 生成权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 生成激活函数
        self.activation_function = lambda x: spc.expit(x)
        pass

    # 训练函数
    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_input = np.dot(self.wih, input)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        # 得到目标值多维数组
        targets = np.array(target_list, ndmin=2).T
        # 计算输出层误差
        output_error = targets - final_output
        # 计算得到隐藏层误差
        hidden_errors = np.dot(self.who.T, output_error)
        # 隐藏层到输出层的权重计算
        self.who += self.lr * np.dot(output_error * final_output * (1.0 - final_output), np.transpose(hidden_output))
        # 输入层到隐藏层的权重计算
        self.wih += self.lr * np.dot(hidden_errors * hidden_output * (1.0 - hidden_output), np.transpose(inputs))
        pass

    # 查询函数
    def query(self, input_list):
        # 拆分为多维数组
        inputs = np.array(input_list, ndmin=2).T

        # 隐藏层,点乘时第一个矩阵的列数目应该等于第二个矩阵的行数目
        hidden_input = np.dot(self.wih, inputs)
        hidden_output = self.activation_function(hidden_input)

        # 输出层
        final_input = np.dot(self.who, hidden_output)
        final_output = self.activation_function(final_input)

        return final_output


n = neuralNetwork(3, 3, 3, 0.3)
print(n.query([1.0, 0.5, -1.5]))
