'''
#### 线性回归+房价与房屋尺寸关系的线性
####线性回归：
- 线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互
依赖的定量关系的一种统计分析方法
- 线性回归是利用线性回归方程的最小平方函数对一个或多个自变量和因变量
之间关系进行建模。这种函数是一个或多个成为回归系数的模型参数的线性组合。
只有一个自变量的情况成为简单回归，大于一个自变量的情况叫多元回归。
#### 线性回归的实际用途
1. 如果目标是预测或者映射，线性回归可以用来对观测数据集的y和x的值
拟合出一个预测模型，当完成这样一个模型以后，对于一个新增的x值，在没有
给定与它相配对的y情况下，可以用这个拟合过的模型预测出一个y值

2.给定一个变量y和一些变量x1，x2，...,xp，这些变量有可能与y相关，
线性回归分析可以用来量化y与xj之间相关性的强度，评估与y不相关的xj，
并识别出哪些xj的子集包含了关于y的冗余信息。
'''

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def Linear_RG():
    dataset_X = []
    dataset_Y = []
    fr = open(r'D:\备份\undergraduate\sklearn学习\回归\prices.txt')
    for line in fr.readlines():
        items = line.strip().split(',')
        dataset_X.append(int(items[0]))
        dataset_Y.append(int(items[1]))
    length = len(dataset_Y)
    dataset_X = np.array(dataset_X).reshape([length, 1])
    print(dataset_X.shape)
    dataset_Y = np.array(dataset_Y).reshape([length, 1])
    print(dataset_Y.shape)
    minX = min(dataset_X)
    maxX = max(dataset_X)
    X = np.arange(minX, maxX).reshape([-1, 1])
    print(X.shape)
    linear = LinearRegression()
    linear.fit(dataset_X, dataset_Y)
    print('Cofficients:', linear.coef_)
    print('intercept:', linear.intercept_)
    result = linear.predict(X)
    plt.scatter(dataset_X, dataset_Y, color='red')
    plt.plot(X, result, color='blue')
    plt.show()


def PolyRG():
    '''
    - 多项式回归，研究一个因变量与一个自变量间多项式的回归分析方法，如果自变量只有一个时，成为一元多项式回归
    ，如果自变量有多个时，成为多元多项式回归
    在一元回归分析中，如果依变量y与自变量x的关系为非线性，但是又找不到适当的函数曲线来拟合，则可以采用一元多项式回归。
    - sklearn中多项式回归
    这里的多项式回归实际上是先将变量x处理成多项式特征，然后使用线性模型学习多项式特征的参数，以达到多项式回归的目的
    :return:
    '''
    datasets_X = []
    datasets_Y = []
    fr = open(r'D:\备份\undergraduate\sklearn学习\回归\prices.txt')
    for line in fr.readlines():
        datasets_X.append(eval(line.strip().split(',')[0]))
        datasets_Y.append(eval(line.strip().split(',')[1]))
    datasets_X = np.array(datasets_X).reshape((-1, 1))
    datasets_Y = np.array(datasets_Y).reshape((-1, 1))
    print(datasets_X.shape, datasets_Y.shape)
    Poly = PolynomialFeatures(degree=3)
    # linear = LinearRegression()
    # linear.fit(datasets_X, datasets_Y)
    X = np.arange(min(datasets_X), max(datasets_X)).reshape([-1, 1])
    # 先构造多项式特征
    X_poly = Poly.fit_transform(datasets_X)  # [x1,x2,x1x2,x1*x1,x1*x2,x1x2]
    linear2 = LinearRegression()
    linear2.fit(X_poly, datasets_Y)
    plt.scatter(datasets_X, datasets_Y, color='red')
    Y = linear2.predict(Poly.fit_transform(X))
    print(X)
    print(Y)
    plt.plot(X, linear2.predict(Poly.fit_transform(X)), color='blue')
    plt.show()


if __name__ == '__main__':
    PolyRG()
