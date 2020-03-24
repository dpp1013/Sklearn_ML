'''
tree.DecisionTreeClassifier
参数
- criterion:用于选择属性的准则,可以传入'gini'代表基尼系数,或者'entropy'代表信息增益
- max_features:表示在决策树结点进行分类时,从多少个特征中选择最优特征.可以设定固定数目,百分比或其他标准.
它的默认值是使用所有特征个数
'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
# 导入计算交叉验证值的函数cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

if __name__ == '__main__':
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    tree = DecisionTreeClassifier(criterion='gini')
    # print(cross_val_score(tree, iris.data, iris.target, cv=10))
    tree.fit(x_train, y_train)
    y_predict = tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict, normalize=True, sample_weight=None)
    print(confusion_matrix(y_test, y_predict, labels=None, sample_weight=None))
    print(accuracy)
    # print(tree.predict(x_test, y_test))
