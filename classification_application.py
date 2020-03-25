import numpy as np
import pandas as pd
import sklearn.impute
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics


# 用pandas进行文件操作
def load_data_pd(feature_path, target_path):
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    df = pd.read_table(feature_path, delimiter=',', na_values='?', header=None)
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    df = imp.fit_transform(df)
    feature = np.concatenate((feature, df))
    df1 = pd.read_table(target_path, delimiter=',', header=None)
    label = np.concatenate((label, df1))
    print(feature.shape)
    print(label.shape)
    return feature, label


# 用numpy 进行文件操作
def load_data(feature_path, target_path):
    '''
    :return:
    读取特征文件列表和标签文件列表中的内容,归并后返回
    '''
    X = None
    Y = []
    file = open(feature_path, 'r')
    label = open(target_path, 'r')
    count = 0
    # feature
    for i in file.readlines():
        count += 1
        if count > 10000:
            count = 0
            break
        x = np.array([float(i) if i != '?' else np.nan for i in i.strip('\n').split(',')])
        x = np.expand_dims(x, axis=0)
        if X is None:
            X = x
        else:
            X = np.vstack((X, x))
    print(X.shape)
    for i in label.readlines():
        count += 1
        if count > 10000:
            break
        if Y is None:
            Y = i
        else:
            Y.append(eval(i))
    Y = np.array(Y)
    print(Y.shape)
    return X, Y


def preprocess(X):
    imputation = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', copy=True)
    X = imputation.fit_transform(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X)
    return X_train_minmax


estimation = [
    ['KNN', KNeighborsClassifier(n_neighbors=5)],
    ['Naiva_bayes', GaussianNB()],
    ['DecisionTreeClassifier', DecisionTreeClassifier(criterion='gini')]
]

if __name__ == '__main__':
    feature_path = r'D:\备份\undergraduate\sklearn学习\分类\dataset\A\A.feature'
    target_path = r'D:\备份\undergraduate\sklearn学习\分类\dataset\A\A.label'
    X, Y = load_data_pd(feature_path, target_path)

    # X, Y = load_data(feature_path, target_path)
    X = preprocess(X)
    X_train, X_text, Y_train, Y_test = train_test_split(X, Y, train_size=7/10, shuffle=True)
    print(X_train.shape)
    print(X_text.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    for i, j in estimation:
        print('模型{}的结果如下:'.format(i))
        model = j
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_text)
        print(metrics.confusion_matrix(Y_test, y_pred))
        print('准确率：', metrics.accuracy_score(Y_test, y_pred))
        print('类别精度：', metrics.precision_score(Y_test, y_pred, average=None))  # 不求平均
        print('宏平均精度：', metrics.precision_score(Y_test, y_pred, average='macro'))
        print('微平均召回率:', metrics.recall_score(Y_test, y_pred, average='micro'))
        print('加权平均F1得分:', metrics.f1_score(Y_test, y_pred, average='weighted'))

