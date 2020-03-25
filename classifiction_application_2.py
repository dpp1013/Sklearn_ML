'''
数据描述:
网易财经上获得的上证指数的历史数据,爬取了20年的上证指数数据
实验目的:根据当前前150天的历史数据,预测当天上证指数的跌涨
注:
如果收盘价高于开盘价则表示涨
'''

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def load_data(path):
    # index_col 是否显示默认索引列
    data = pd.read_csv(path, parse_dates=[0], index_col=0)
    data.sort_values(by='日期', axis=0, ascending=True, inplace=True)
    dayfeature = 150
    # 总特征
    featurenum = 5 * dayfeature  # 751
    # 构造输入数据
    x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))
    y = np.zeros((data.shape[0] - dayfeature, 1))
    # print(x.shape)
    # print(y.shape)
    for i in range(0, data.shape[0] - dayfeature):
        a = np.array(data[i:i + dayfeature][['收盘价', '开盘价']])
        # print(a.shape)
        x[i, 0:featurenum] = np.array(data[i:i + dayfeature][['收盘价', '最高价', '开盘价', '最低价', '成交量']]).reshape(
            (1, featurenum))
        # 收盘价
        x[i, featurenum] = np.array(data.iloc[i + dayfeature, 2])
    for i in range(0, data.shape[0] - dayfeature):
        # 如果收盘价>=开盘价 说明涨了
        if data.iloc[i + dayfeature, 2] >= data.iloc[i + dayfeature, 5]:
            y[i, 0] = 1
        else:
            y[i, 0] = 0
    return x,y


if __name__ == '__main__':
    path = r'D:\备份\undergraduate\sklearn学习\分类\stock\000777.csv'
    X, Y = load_data(path)
    clf = SVC(kernel='rbf')
    result = []
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=8 / 10, shuffle=True)
        clf.fit(x_train, y_train)
        result.append(np.mean(y_test == clf.predict(x_test)))
    print('SVM classifier accuracy:')
    print(result)
