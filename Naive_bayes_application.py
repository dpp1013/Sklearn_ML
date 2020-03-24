'''
Naive_bayes_application
'''
import numpy as np
from sklearn.naive_bayes import GaussianNB
if __name__ == '__main__':

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    Y = np.array([1, 1, 1, 2, 2, 2])
    bayes = GaussianNB()
    bayes.fit(X, Y)
    print(bayes.predict([[-1,-2],[5,6]]))
