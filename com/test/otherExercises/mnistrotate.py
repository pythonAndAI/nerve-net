import numpy as np
from com.test.nerveNetwork.NumberOneNerveNetwork import neuralNetwork
import scipy.ndimage.interpolation

def test():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learningrate = 0.3
    mnist = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learningrate)

    '''
    训练
    '''
    train_files = open("E:\Alls\软件\MNIST\mnist_train.csv", "r")
    train_list = train_files.readlines()
    # print("train_list size is:", len(train_list))
    train_files.close()

    for train_data in train_list:
         value_spilt = train_data.split(",")
         input_list = (np.asfarray(value_spilt[1:]) / 255.0 * 0.99) + 0.01
         target_list = np.zeros(output_nodes) + 0.01
         target_list[int(value_spilt[0])] = 0.99
         mnist.train(input_list, target_list)

         inputs_plusx_img = scipy.ndimage.interpolation.rotate(input_list.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
         mnist.train(inputs_plusx_img.reshape(784), target_list)
         # rotated clockwise by x degrees
         inputs_minusx_img = scipy.ndimage.interpolation.rotate(input_list.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
         mnist.train(inputs_minusx_img.reshape(784), target_list)

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
        test_input_list = (np.asfarray(test_split[1:]) / 255.0 * 0.09) + 0.01
        final_list = mnist.query(test_input_list)
        test_zero = int(test_split[0])
        label = np.argmax(final_list)
        if label == test_zero:
            scorecard.append(1)
        else:
            scorecard.append(0)
    score_array = np.asarray(scorecard)
    score = score_array.sum() / score_array.size
    return score

if __name__ == "__main__":
    for i in range(10):
        print("第", i, "次的准确率为:", test())