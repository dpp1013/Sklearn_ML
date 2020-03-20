#### 数据说明

| 1 | 2 | 3-15 | 16-18 | 29-41 |
| :-----| :---- | :---- | :----|:---- |
| 时间戳 | 心率 | 传感器1 | 传感器2 |传感器3|
#####KNN
from sklearn.neighbors import KNeighborsClassifier

参数:

- n_neighbors:分类器中k的大小
- weights:设置选中k个点对分类结果影响的权重(默认为平均权重'uniform',可以选择'distance'代表越近的点权重
越高,或者传入自己编写的以距离为参数的权重计算函数)

#####决策树
from sklearn import DecisionTreeClassifier

参数
- criterion:用于选择属性的准则,可以传入'gini'代表基尼系数,或者'entropy'代表信息增益
- max_features:表示在决策树结点进行分类时,从多少个特征中选择最优特征.可以设定固定数目,百分比或其他标准.
它的默认值是使用所有特征个数