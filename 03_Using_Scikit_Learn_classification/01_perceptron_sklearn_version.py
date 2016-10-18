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


def read_csv_pd():
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/'
    #        'machine-learning-databases/iris/iris.data', header=None)
    df = pd.read_csv(data_csv_path, header=None)

    #df.to_csv(filepath+'/'+source_folder+'/iris.csv',index=0,header=False)

    print(df.tail())
    print(df.shape)
    return df



#A function for plotting decision regions
def plot_decision_regions(X, y, classifier, resolution=0.02):
    from matplotlib.colors import ListedColormap
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

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)



def plot_2d_data(X,y):
    ######################
    # plot to take a look,Plotting the Iris data
    ####################

    # plot data
    plt.scatter(X[:50, 0], X[:50, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],
                color='blue', marker='x', label='versicolor')

    plt.xlabel('petal length [cm]')
    plt.ylabel('sepal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./iris_1.png', dpi=300)
    plt.show()
    return X,y





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
    print('Class labels:', np.unique(y))




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




    ###############
    # Training a perceptron via scikit-learn
    #####################
    from sklearn.linear_model import Perceptron

    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)

    ######################
    #ready to go testing
    ######################
    print(y_test.shape)
    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    #################
    # evaulation
    #################
    from sklearn.metrics import accuracy_score

    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


    ##########
    # plot result
    ###########
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    plot_decision_regions(X=X_combined_std, y=y_combined,
                          classifier=ppn, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
    plt.show()

    return 0


main()
