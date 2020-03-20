'''
tree.DecisionTreeClassifier
参数
- criterion:用于选择属性的准则,可以传入'gini'代表基尼系数,或者'entropy'代表信息增益
- max_features:表示在决策树结点进行分类时,从多少个特征中选择最优特征.可以设定固定数目,百分比或其他标准.
它的默认值是使用所有特征个数
'''
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris = load_iris()
    x_train, y_train, x_text, y_text = train_test_split(iris, iris.target, test_size=0.3)
    tree = DecisionTreeClassifier(criterion='gini', max_features=30)
    tree.fit(x_train, y_train)
    predict_target = tree.predict(x_text)
    print(sum(predict_target == y_text))

