import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        X = np.array(X)
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)


###########################################################################
# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# PCA decomposition to get two feature
X = df.iloc[0:100, [0,1,2,3]].values
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)


ppn = Perceptron(epochs=10, eta=0.1)

ppn.train(X, y)
plot_decision_regions(X, y, clf=ppn)
plt.title(' setosa VS versicolor ')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.title(' setosa VS versicolor ')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()



###########################################################################
# versicolor and virginica
y2 = df.iloc[50:150, 4].values
y2 = np.where(y2 == 'Iris-virginica', -1, 1)

# PCA decomposition to get two feature
X = df.iloc[50:150, [0,1,2,3]].values
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

ppn = Perceptron(epochs=25, eta=0.01)
ppn.train(X, y2)

plot_decision_regions(X, y2, clf=ppn)
plt.title(' versicolor VS virginica ')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.title(' versicolor VS virginica ')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

###########################################################################
# setosa and virginica
#y2 = df.iloc[50:150, 4].values
y2 = df.loc[(df[4] == 'Iris-virginica') | (df[4] == 'Iris-setosa') , 4].values
y2 = np.where(y2 == 'Iris-virginica', -1, 1)

# PCA decomposition to get two feature
X = df.loc[(df[4] == 'Iris-virginica') | (df[4] == 'Iris-setosa'), [0,1,2,3]].values
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

ppn = Perceptron(epochs=25, eta=0.01)
ppn.train(X, y2)

plot_decision_regions(X, y2, clf=ppn)
plt.title(' setosa VS virginica ')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.title(' setosa VS virginica ')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()