import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def read_data():

    #df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    #read from csv
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/wine.csv'
    df_wine = pd.read_csv(data_csv_path, header=None)


    #save csv
    '''
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/wine.csv'
    df_wine.to_csv(data_csv_path,index=0,header=False)
    '''

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    print('Class labels', np.unique(df_wine['Class label']))
    print(df_wine.head())
    print(df_wine.shape)
    return df_wine

def plot_decision_regions(X, y, classifier, resolution=0.02):

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

def main():
    df_wine=read_data()
    ##############
    #Partitioning a dataset in training and test sets
    #############
    from sklearn.cross_validation import train_test_split

    X = df_wine.iloc[:, 1:].values
    y = df_wine.iloc[:, 0].values
    #separate the features to X, label to y in numpy type
    #print(X)
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    #print(X_train)
    #print(y_train)

    ################
    # feature standardization
    ################
    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)

    #Data ready

    #######################
    #hand made: Supervised data compression via linear discriminant analysis
    #######################



    ################################
    # prepare :Computing the scatter matrices (1.with in class, 2.between-class)
    ################################

    # Calculate the mean vectors for each class:
    np.set_printoptions(precision=4)

    mean_vecs = []
    #will get 3 mean vectors, because we have 3 class of wine
    for label in range(1,4):
        mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
        print('MV %s: %s\n' %(label, mean_vecs[label-1]))
    '''
    MV 1: [ 0.9259 -0.3091  0.2592 -0.7989  0.3039  0.9608  1.0515 -0.6306  0.5354
      0.2209  0.4855  0.798   1.2017]

    MV 2: [-0.8727 -0.3854 -0.4437  0.2481 -0.2409 -0.1059  0.0187 -0.0164  0.1095
     -0.8796  0.4392  0.2776 -0.7016]

    MV 3: [ 0.1637  0.8929  0.3249  0.5658 -0.01   -0.9499 -1.228   0.7436 -0.7652
      0.979  -1.1698 -1.3007 -0.3912]
    '''


    #Compute the within-class (class1 or class2 ..) scatter matrix:
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.zeros((d, d)) # scatter matrix for each class
        for row in X[y == label]:
            row, mv = row.reshape(d, 1), mv.reshape(d, 1) # make column vectors
            class_scatter += (row-mv).dot((row-mv).T)
        S_W += class_scatter                             # sum class scatter matrices

    print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
    '''
    Within-class scatter matrix: 13x13
    '''

    #Better: covariance matrix since classes are not equally distributed:
    print('Class label distribution: %s' % np.bincount(y_train)[1:])
    '''
    # three class data are not distribution equally well. ex: 30=30=30
    Class label distribution: [40 49 35]
    '''


    #so we Scaled within-class scatter matrix
    d = 13 # number of features
    S_W = np.zeros((d, d))
    for label,mv in zip(range(1, 4), mean_vecs):
        class_scatter = np.cov(X_train_std[y_train==label].T)
        S_W += class_scatter
    print('Scaled within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))

    #Compute the between-class scatter matrix:
    mean_overall = np.mean(X_train_std, axis=0)
    d = 13 # number of features
    S_B = np.zeros((d, d))
    for i,mean_vec in enumerate(mean_vecs):
        n = X[y==i+1, :].shape[0]
        mean_vec = mean_vec.reshape(d, 1) # make column vector
        mean_overall = mean_overall.reshape(d, 1) # make column vector
        S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

    print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))


    #Solve the generalized eigenvalue problem for the matrix
    # the same step as PCA hand made
    eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    #Sort eigenvectors in decreasing order of the eigenvalues:
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues

    print('Eigenvalues in decreasing order:\n')
    for eigen_val in eigen_pairs:
        print(eigen_val[0])

    '''
    Eigenvalues in decreasing order:

    643.015384346 ---->None Zero
    225.086981854 ---->None Zero
    7.77211874802e-14----> ~=0, because the numpy floating operation
    6.62429382151e-14----> ~=0
    5.56337879014e-14----> ~=0
    5.56337879014e-14----> ~=0
    2.39743505423e-14----> ~=0
    1.215658955e-14----> ~=0
    1.215658955e-14----> ~=0
    5.0217085079e-15----> ~=0
    4.52734964452e-15----> ~=0
    4.52734964452e-15----> ~=0
    0.0
    '''



    #plot out
    tot = sum(eigen_vals.real)
    discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
    cum_discr = np.cumsum(discr)

    plt.bar(range(1, 14), discr, alpha=0.5, align='center',
            label='individual "discriminability"')
    plt.step(range(1, 14), cum_discr, where='mid',
             label='cumulative "discriminability"')
    plt.ylabel('"discriminability" ratio')
    plt.xlabel('Linear Discriminants')
    plt.ylim([-0.1, 1.1])
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/lda1.png', dpi=300)
    plt.show()

    #you can check the pic to find, the fisrt two item can
    # discriminability all the wine training data



    #we chose first two Eigenvalues and select the eigen vector
    # to became a transform matrix
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
                      eigen_pairs[1][1][:, np.newaxis].real))
    print('Matrix W:\n', w)

    '''
    Matrix W:
     [[-0.0707 -0.3778]
     [ 0.0359 -0.2223]
     [-0.0263 -0.3813]
     [ 0.1875  0.2955]
     [-0.0033  0.0143]
     [ 0.2328  0.0151]
     [-0.7719  0.2149]
     [-0.0803  0.0726]
     [ 0.0896  0.1767]
     [ 0.1815 -0.2909]
     [-0.0631  0.2376]
     [-0.3794  0.0867]
     [-0.3355 -0.586 ]]
     '''

    ################
    # project data to sub space
    ###############

    X_train_lda = X_train_std.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_lda[y_train==l, 0],
                    X_train_lda[y_train==l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig('./figures/lda2.png', dpi=300)
    plt.show()

    #you saw the pic, and found it could be separate by linear


    return 0
main()
