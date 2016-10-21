import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
print(filepath)
data_csv_path=filepath+'/'+source_folder+'/iris.csv'



################################
#implement the model class
###############################
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


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



def plot_2d_data(df):
    ######################
    # plot to take a look,Plotting the Iris data
    ####################


    # select setosa and versicolor
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # extract sepal length and petal length
    X = df.iloc[0:100, [0, 2]].values

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

##############
#Training the perceptron model
############
def train_perceptron_model(X,y):
    #new a object
    ppn = Perceptron(eta=0.1, n_iter=10)

    ppn.fit(X, y)

    #show the error history
    #it shows the model converge at 6th round.
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

    plt.tight_layout()
    # plt.savefig('./perceptron_1.png', dpi=300)
    plt.show()
    #return the model object
    return ppn

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

def plot_decision_plan(ppn,X,y):
    plot_decision_regions(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./perceptron_2.png', dpi=300)
    plt.show()
    return 0


def plot_leraning_curve(X, y):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Sum-squared-error)')
    ax[0].set_title('Adaline - Learning rate 0.01')

    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Sum-squared-error')
    ax[1].set_title('Adaline - Learning rate 0.0001')

    plt.tight_layout()
    # plt.savefig('./adaline_1.png', dpi=300)
    plt.show()
    return 0


def standardizing_features(X):
    # standardize features
    X_std = np.copy(X)
    X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
    X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    return X_std

def train_with_std_x_and_its_lernaing_curve(X_std, y):
    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    plot_decision_regions(X_std, y, classifier=ada)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.savefig('./adaline_2.png', dpi=300)
    plt.show()

    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')

    plt.tight_layout()
    # plt.savefig('./adaline_3.png', dpi=300)
    plt.show()
    return 0


def main():
    data_df=read_csv_pd()
    X,y=plot_2d_data(data_df)
    #before the Standardizing, show with different learning spped
    # it show, when learning speed = 0.01 is too fast to converge
    plot_leraning_curve(X, y)

    #standardize features
    X_std=standardizing_features(X)

    #use the X_std to model again, and use learning speed 0.01 again.
    # this time, 0.01 will converge smoothly
    train_with_std_x_and_its_lernaing_curve(X_std, y)


    return 0


main()
