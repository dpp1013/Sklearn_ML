'''
聚类：通过聚类了解1999年各省的消费水平在国内的情况
技术路线：sklearn.cluster.Kmeans
调用K-means方法所需要参数：
n_cluster:用于指定聚类中心的个数
init：初始化聚类中心的初始化方法
max_iter:最大的迭代次数
一般只用给出n_cluster即可，init默认是k-means++,max_iter默认是300
程序中其他参数
data：加载的数据
label:聚类后各数据所属的标签
axis：按行求和
fit_predict():计算簇中心以及为簇分配序号
'''
import numpy as np

from sklearn.cluster import KMeans


def loadData(path):
    file = open(path, 'r')
    data = []
    cityname = []
    for line in file.readlines():
        cityname.append(line.split(',')[0])
        data.append([float(i) for i in line.strip('\n').split(',')[1:]])
    return data, cityname


if __name__ == '__main__':
    data, cityname = loadData(r'D:\备份\undergraduate\sklearn学习\聚类\31省市居民家庭消费水平-city.txt')
    km = KMeans(n_clusters=3)  # 创建实例
    label = km.fit_predict(data)  # 调用方法
    print(dir(label))

    expense = np.sum(km.cluster_centers_, axis=1)  # 按簇进行中心消费计算
    # 存储聚类结果
    CityCluster = [[], [], []]
    for i in range(len(cityname)):
        CityCluster[label[i]].append(cityname[i])
    for i in range(len(CityCluster)):
        print('Expense:%.2f' % expense[i])
        print(CityCluster[i])
