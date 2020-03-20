'''
通过DBSCAN聚类，分析学生上网时间和上网时长的模式
DBSCAN的主要参数
- eps：两个样本被看作邻居节点的最大距离
- min_sample:簇的样本数
- metric：距离计算方式

'''
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def getData():
    mac2id = {}
    onlineitems = []
    f = open(r'D:\备份\undergraduate\sklearn学习\聚类\学生月上网时间分布-TestData.txt', encoding='utf-8')
    for line in f.readlines():
        mac = line.split(',')[2]
        onlinetime = int(line.split(',')[6])
        starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
        if mac not in mac2id:
            mac2id[mac] = [starttime, onlinetime]
            onlineitems.append([starttime, onlinetime])
        else:
            onlineitems[mac2id[mac]] = [(starttime, onlinetime)]
    real_X = np.array(onlineitems)
    return real_X


def startTime(X):
    db = DBSCAN(eps=0.01, min_samples=20).fit(X)
    labels = db.labels_
    print('label:')
    print(labels)
    # label 为-1表示噪声数据
    raito = len(labels[labels[:] == -1]) / len(labels)
    print('Noise ratio:{}'.format(raito, '.2%'))
    # set(labels) set去重
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters:%d' % n_clusters_)
    print('Silhouette Coefficient:%0.3f' % metrics.silhouette_score(X, labels))

    for i in range(n_clusters_):
        print('Cluster:', i, ':')
        print(list(X[labels == i].flatten()))


def onlineTime(X):
    db = DBSCAN(eps=0.14, min_samples=10).fit(X)
    labels = db.labels_
    print('label:')
    print(labels)
    # label 为-1表示噪声数据
    raito = len(labels[labels[:] == -1]) / len(labels)
    print('Noise ratio:{}'.format(raito, '.2%'))
    # set(labels) set去重
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Estimated number of clusters:%d' % n_clusters_)
    print('Silhouette Coefficient:%0.3f' % metrics.silhouette_score(X, labels))

    for i in range(n_clusters_):
        print('Cluster:', i, ':')
        count = len(X[labels == i])
        mean = np.mean(X[labels == i][:, 1])
        std = np.std(X[labels == i][:, 1])
        print('\t number of sample:', count)
        print('\t mean of sample:', format(mean, '.1f'))
        print('\t std of sample:', format(std, '.1f'))


if __name__ == '__main__':
    Data = getData()
    # print(Data)
    X = Data[:, 0:1]
    # startTime(X)
    # 数据分布不明显
    Y = np.log(1 + Data[:, 1:2])
    onlineTime(Y)
    # plt.hist(Y, 24)
    # plt.show()
