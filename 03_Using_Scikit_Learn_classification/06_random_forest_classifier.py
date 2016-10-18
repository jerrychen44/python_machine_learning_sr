import os
import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
print(filepath)
data_csv_path=filepath+'/'+source_folder+'/iris.csv'

from sklearn import datasets



#####################
#loading data set
# ref: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
# Attribute Information:
#   1. sepal length in cm
#   2. sepal width in cm
#   3. petal length in cm
#   4. petal width in cm
#   5. class:
#      -- Iris Setosa
#      -- Iris Versicolour
#      -- Iris Virginica
##########################

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')


def gini(p):
    return (p)*(1 - (p)) + (1-p)*(1 - (1-p))

def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

def error(p):
    return 1 - np.max([p, 1 - p])


def visualized_gini_entropy_error():
    x = np.arange(0.0, 1.0, 0.01)

    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [error(i) for i in x]


    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                      ['Entropy', 'Entropy (scaled)',
                       'Gini Impurity', 'Misclassification Error'],
                      ['-', '-', '--', '-.'],
                      ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)



    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fancybox=True, shadow=False)

    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.tight_layout()
    #plt.savefig('./figures/impurity.png', dpi=300, bbox_inches='tight')
    plt.show()
    return 0

def main():


    ###########
    # Get data
    ###########
    #data_df=read_csv_pd()
    iris = datasets.load_iris()
    #print(iris)
    X = iris.data[:, [2, 3]] #get the feature 2 petal length (cm), and feature 3 petal width (cm)
    #print(X)
    y = iris.target


    #########################
    # seperte testing, traing data to train
    # Splitting data into 70% training and 30% test data:
    ######################
    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


    ##################
    # Standardizing the features:
    ##################
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    ################
    # Model
    #################
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(criterion='entropy',
                                    n_estimators=10,#use 10 desicion tree
                                    random_state=1,
                                    n_jobs=2)#use 2 cpu
    forest.fit(X_train, y_train)

    plot_decision_regions(X_combined, y_combined,
                          classifier=forest, test_idx=range(105,150))

    plt.xlabel('petal length [cm]')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('./figures/random_forest.png', dpi=300)
    plt.show()


    return 0


main()
