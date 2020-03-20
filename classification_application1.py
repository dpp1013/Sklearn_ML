'''
sklearn中的k近邻分类器
sklearn.neighbors.KNeighborsClassifier创建一个k近邻分类器,主要参数
n_neighbors:分类器中k的大小
weights:设置选中k个点对分类结果影响的权重(默认为平均权重'uniform',可以选择'distance'代表越近的点权重
越高,或者传入自己编写的以距离为参数的权重计算函数)
'''
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':

    X = [[0], [1], [2], [3]]
    y = [0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
