'''
一般的线性回归问题，参数的求解采用的是最小二乘法，其目标函数如下：
sklearn中的岭回归的参数：
1. alpha：正则化因子，对应于损失函数的a
2. fit_intercept:表示是否计算借据
3. solver:设置计算参数的方法，可选参数'auto','svd','sag'等
'''

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取csv文件，为array对象
    data = np.array(pd.read_csv(r'D:\备份\undergraduate\sklearn学习\回归\岭回归.csv'))
    print(data.shape)
    X = data[:, 1:4]
    Y = data[:, 5:6]
    print(Y)
    print(X.shape, Y.shape)
    poly = PolynomialFeatures(6)
    X = poly.fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.7, random_state=0)
    clf = Ridge(alpha=1.0, fit_intercept=True)
    clf.fit(train_x, train_y)
    # 计算测试集计算回归曲线的拟合优度，clf.score
    # 拟合优度，用于评价拟合好坏，最大为1，无最小值，党对所有输入都输出同一个值时，拟合优度为0
    start = 200
    end = 300
    y_pre = clf.predict(X)
    time = np.arange(start, end)
    print(Y[start:end])
    plt.plot(time, Y[start:end], 'b', label='real')
    plt.plot(time, y_pre[start:end], 'r', label='predict')
    plt.legend(loc='upper left')
    plt.show()
