'''
sklearn中的主成分分析
sklearn.decomposition.PCA加载PCA进行降维，主要参数有：
n_components：指定主成分的个数，即降维后数据的维度
svd_solver:设置特征值分解的方法，默认为‘auto',其他可以选择为full，arpack，randomized
PCA实现高维数据可视化
fit适配的过程,不是train,最后只是得到一个统一的转换的规则模型
transform:将数据进行转换,比如数据的归一化标准化,将测试数据按照训练数据同样的模型进行转换,得到的特征向量
fit_transform:可以看做是fit和transform的结合,如果训练阶段使用fit_transofrom,则在测试阶段只需要对测试样本进行transfrom

'''
# PCA实现高维数据可视化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)  # 降维后的主成分为2
# fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
reduced_X = pca.fit_transform(X)  # 注意此处不是fit_predict
print(len(reduced_X))
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()

