import numpy as np
from com.test.mnistTest import mnistTest3 as mt3
from com.test.mnistTest import mnistTest2 as mt2
import matplotlib.pyplot as plt
import time

'''
调整神经网络的学习率、训练次数和隐藏层节点数量来改进神经网络识别的准确度
'''

class mnist():

    def __init__(self, isitfull, testtype, draw):
        self.isf = isitfull
        self.fig = draw
        self.tt = testtype

    '''
    根据isitfull值来确认运行的数据集是全量还是少量
    '''
    def type_confirm(self):
        if self.isf:
            self.tt = mt3
        else:
            self.tt = mt2

    '''
    调整学习率
    '''
    def learn_num(self):
        scores = []
        learnings = []
        for lr in range(10):
            if lr == 0:
                lr = 0.01
                learnings.append(lr)
            else:
                lr = lr / 10
                learnings.append(lr)
            sroce = self.tt.test(100, lr, 1)
            scores.append(sroce)
        self.drawing_statistics(learnings, scores, "learn")

    '''
    调整训练次数
    '''
    def train_num(self):
        trains = []
        scores = []
        for i in range(1, 11, 1):
            sroce = self.tt.test(100, 0.3, i)
            trains.append(i)
            scores.append(sroce)
        self.drawing_statistics(trains, scores, "train")

    '''
    调整隐藏层的节点数量
    '''
    def hidden_num(self):
        scores = []
        hidden = []
        for hd in range(0, 600, 100):
            if hd == 0:
                hd = 10
                hidden.append(hd)
            else:
                hidden.append(hd)
            sroce = self.tt.test(hd, 0.3, 1)
            scores.append(sroce)
        self.drawing_statistics(hidden, scores, "hidden")

    '''
    图形化表示
    '''
    def drawing_statistics(self, horizontal, vertical, type):
        horizontal_array = np.asarray(horizontal)
        vertical_array = np.asarray(vertical)
        if type == "learn":
            ax = self.fig.add_subplot(2, 2, 1)
            plt.xlabel("learningrate")
        elif type == "train":
            ax = self.fig.add_subplot(2, 2, 2)
            plt.xlabel("trainnumber")
        else:
            ax = self.fig.add_subplot(2, 1, 2)
            plt.xlabel("hiddennumber")
        ax.plot(horizontal_array, vertical_array)
        plt.ylabel("score")

if __name__ == "__main__":
    print('运算开始时间:',time.strftime('%H:%M:%S', time.localtime()))
    fig = plt.figure(num=3, figsize=(15, 8), dpi=80)
    mnist = mnist(False, testtype=None, draw=fig)
    mnist.type_confirm()
    mnist.learn_num()
    mnist.train_num()
    mnist.hidden_num()
    print('运算结束时间:', time.strftime('%H:%M:%S', time.localtime()))
    plt.show()
    plt.close()


