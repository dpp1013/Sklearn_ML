'''
任务：
利用sklearn来训练一个简单的全连接神经网络，即多层感知器，用于识别DBRHD手写数字
input: 32*32 =1024
output:'one-hot vectors'[1,0,0,0,0,0,0,0,0]
'''
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def img2vec(path):
    img = np.zeros([1024], int)
    file = open(path)
    data = file.readlines()
    for i in range(32):
        for j in range(32):
            img[i * 32 + j] = data[i][j]
    return img


def load_data(path):
    filename = os.listdir(path)
    filenum = len(filename)
    print(filenum)
    dataset_X = np.zeros([filenum, 1024])
    dataset_Y = np.zeros([filenum, 10])
    for i in range(filenum):
        dataset_X[i][:] = img2vec(os.path.join(path, filename[i]))
        num = eval(filename[i].split('_')[0])
        dataset_Y[i][num] = 1.0
    return dataset_X, dataset_Y


if __name__ == '__main__':
    path1 = r'D:\备份\undergraduate\sklearn学习\手写数字\digits\trainingDigits'
    path2 = r'D:\备份\undergraduate\sklearn学习\手写数字\digits\testDigits'
    dataset_X, dataset_Y = load_data(path1)
    # train_x, test_x, train_y, test_y = train_test_split(dataset_X, dataset_Y, train_size=0.7, shuffle=True)
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver='adam', learning_rate_init=0.0001,
                        max_iter=2000 )
    clf.fit(dataset_X, dataset_Y)
    dataset, label = load_data(path2)
    res = clf.predict(dataset)
    error_num = 0
    num = len(dataset)
    for i in range(num):
        if np.sum(res[i] == label[i]) < 10:
            error_num += 1
    print('total num:', num, 'wrong num:', error_num, 'wrongRate:', error_num / float(num))
