import numpy as np
import imageio
import scipy.misc
import glob
from com.test.nerveNetwork.NumberOneNerveNetwork import neuralNetwork
import os

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

    image_path = "E:\\Alls\\软件\\MNIST\\makeyourownneuralnetwork-master\\my_own_images\\2828_my_own_?.png"
    #image_log = "E:\\Alls\\软件\\MNIST\\makeyourownneuralnetwork-master\\my_own_images\\mnist.log"
    #image_log_file = open(image_log, "a")
    our_own_dataset = []
    '''
    读取自制的图片的像素，并组装
    '''
    for image in glob.glob(image_path):
        label = int(image[-5:-4])
        image_array = scipy.misc.imread(image, flatten = True)
        #image_array2 = imageio.imread(image_path, as_gray=True)
        image_data = 255.0 - image_array.reshape(784)
        image_data = (image_data / 255.0 * 0.99) + 0.01
        #image_log_file.write(image + " image_data is:" + str(image_data) + "\n")
        record = np.append(label, image_data)
        #image_log_file.write(image + " record is:" + str(record) + "\n")
        our_own_dataset.append(record)
        #image_log_file.write(image + " our_own_dataset is:" + str(our_own_dataset) + "\n")
    #image_log_file.close()
    '''
    用自制的图片测试
    '''
    for our_own_data in our_own_dataset:
        image_input = our_own_data[1:]
        output = mnist.query(image_input)
        image_zero = our_own_data[0]
        label = np.argmax(output)
        print(image_zero, label)
        if label == image_zero:
            print("success")
        else:
            print("fail")