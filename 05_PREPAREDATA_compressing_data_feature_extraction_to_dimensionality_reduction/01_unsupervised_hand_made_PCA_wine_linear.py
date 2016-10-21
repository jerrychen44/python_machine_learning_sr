import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt



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

    #################
    # hand made for PCA
    #################
    #Eigendecomposition of the covariance matrix.
    cov_mat = np.cov(X_train_std.T)
    print('\ncovariance matrix \n%s' % cov_mat)
    print(cov_mat.shape)
    #np.linalg.eig , sometimes will return a veray complex values
    #np.linalg.eigh, must return the eigenvaluse
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

    print('\nEigenvalues \n%s' % eigen_vals)
    print(eigen_vals.shape)
    '''
    Eigenvalues
    [ 4.8923083   2.46635032  1.42809973  1.01233462  0.84906459  0.60181514
      0.52251546  0.08414846  0.33051429  0.29595018  0.16831254  0.21432212
      0.2399553 ]
    (13,)
    '''
    ###########
    # take a look "explained variance"
    ##########
    #prepare the data
    tot = sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    #plot out
    plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(1, 14), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('./figures/pca1.png', dpi=300)
    plt.show()

    #look at the pic, you can find the first principal componects can explained 40% varance
    # and second PC can explained 20%, so we choose these two PC to use. (just for example)
    # how many PC be selected need to be evaluation.




    #######################
    # Feature transformation
    # since we try to use the first and second PC to be a new axis, so
    # right now we can transform origianl wine feature to new feature
    #########################
    # Make a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(reverse=True)

    #we select the first two eigen_pairs
    #and make a 13x2 projecton matrix w
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    '''
    Matrix W:
     [[ 0.14669811  0.50417079]
     [-0.24224554  0.24216889]
     [-0.02993442  0.28698484]
     [-0.25519002 -0.06468718]
     [ 0.12079772  0.22995385]
     [ 0.38934455  0.09363991]
     [ 0.42326486  0.01088622]
     [-0.30634956  0.01870216]
     [ 0.30572219  0.03040352]
     [-0.09869191  0.54527081]
     [ 0.30032535 -0.27924322]
     [ 0.36821154 -0.174365  ]
     [ 0.29259713  0.36315461]]
    '''
    #try out to transform a data 1x13 to new 1x2 feature space throuth w
    print(X_train_std[0])
    print(X_train_std[0].dot(w))


    # we transform all origianl 123x13 data to 123x2 through this w
    X_train_pca = X_train_std.dot(w)

    #plot out the data after transformed
    #you can found the data along the PC1 is more separate(see the data variance in X axis) then PC2 (Y).
    #it is what we say before , that th PC1 has 40% variance.
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o'] # three kinds of wine, but it just for showing, PCA is unsupervised method.
    #x is the transformed data

    for l, c, m in zip(np.unique(y_train), colors, markers):
        plt.scatter(X_train_pca[y_train==l, 0],
                    X_train_pca[y_train==l, 1],
                    c=c, label=l, marker=m)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('./figures/pca2.png', dpi=300)
    plt.show()


    return 0

main()
