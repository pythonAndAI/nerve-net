#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt


class numpyTest:
    def test(self):
        # 创建一个3*2的数组，默认值为0
        a = np.zeros([3, 2])
        a1 = np.zeros([8])
        a[1, 1] = 1
        a[2, 1] = 2
        # 绘制数组
        plt.imshow(a, interpolation="nearest")
        plt.show()
        print("a===>", a)
        print("a1===>", a1)

        # 创建一个随机数
        b = np.random.rand(3, 3) - 0.5
        c = np.random.normal(0.0, pow(3, -0.5), (3, 3))
        j = c.T
        print("b===>", b)
        print("c===>", c)
        print("j===>", j)

        # 把输入的数组拆分成多维数组
        d = np.array([1, 2, 3], ndmin=2).T
        e = np.array([1, 2, 3], ndmin=2)
        f = np.array([1, 2, 3])
        f1 = np.asarray([1, 2, 3])
        print("d===>", d)
        print("e===>", e)
        print("f===>", f)
        print("f1===>", f1)
        # 数组转换
        g = [1, 2, 3]
        h = np.array(g, ndmin=2).T
        i = np.transpose(h)
        print("g===>", g)
        print("h===>", h)
        print("i===>", i)

        # 字符文本转换成矩阵
        j = '1,2,3,4,5,6,7,8,9'
        k = j.split(",")
        l = np.asfarray(k).reshape((3, 3))
        m = np.asfarray(k)
        print(k)
        print(l)
        print(m)
        print("11")
        pass


t = numpyTest()
t.test()
