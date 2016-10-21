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



    #################
    # PCA from sklearn
    #################
    from sklearn.decomposition import PCA

    pca = PCA(n_components=None)# set to None to show all pc
    X_train_pca = pca.fit_transform(X_train_std) #which means will still keep 13 dimetion.No dim reduce happened.


    #show all the pc variance ratio ,explained variance
    print(pca.explained_variance_ratio_)
    '''
    [ 0.37329648  0.18818926  0.10896791  0.07724389  0.06478595  0.04592014
      0.03986936  0.02521914  0.02258181  0.01830924  0.01635336  0.01284271
      0.00642076]
    '''
    #plot
    plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.show()



    ###############
    # transform , 124x13 -> 124x2
    #################
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    #plot out the transformed result. (13 feature -> 2 dimetion)
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1])
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()


    #################
    # Training logistic regression classifier using the first 2 principal components.
    # you can chose any classification model to do the classify
    # we try the logistic regression to classify
    ##################
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr = lr.fit(X_train_pca, y_train)

    #plot the result
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('./figures/pca3.png', dpi=300)
    plt.show()


    ##############
    # try the testing set
    ##############
    plot_decision_regions(X_test_pca, y_test, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.savefig('./figures/pca4.png', dpi=300)
    plt.show()

    return 0

main()
