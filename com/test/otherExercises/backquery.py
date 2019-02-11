import numpy as np
from com.test.nerveNetwork.NumberOneNerveNetwork import neuralNetwork
import matplotlib.pyplot as plt

if __name__ == '__main__':

    mnist_file = open('E:\Alls\软件\MNIST\mnist_train.csv', 'r')
    mnist_list = mnist_file.readlines()
    mnist_file.close()

    '''
    mnist训练数据集训练神经网络
    '''
    mnist = neuralNetwork(784, 100, 10, 0.3)
    for mnist_data in mnist_list:
        all_value = mnist_data.split(",")
        input_list = (np.asfarray(all_value[1:]) / 255.0 * 0.99) + 0.01
        target_list = np.zeros(10) + 0.01
        target_list[int(all_value[0])] = 0.99
        mnist.train(input_list, target_list)

    '''
    反向测试神经网络
    '''
    for i in range(10):
        target = np.zeros(10) + 0.01
        target[i] = 0.99
        inputs = mnist.backquery(target)
        # plt.imshow(inputs.reshape(28, 28), cmap='Greys', interpolation='None')
        # plt.show()

        '''
        用反向查询出的数据先训练在测试，准确率50%左右。因为inputs已经为多维数组，所以不需要再转换
        '''
        mnist.train2(inputs, target)
        final = mnist.query2(inputs)
        label = np.argmax(final)
        if label == i:
            print("i is:", i, "success")
        else:
            print("i is:", i, "label is:", label, "fail")