from sklearn.base import clone
import pandas as pd
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os
#######################
# Hand made "sequential feature selection" (sklearn doesn't have it)
# using SBS algo.
######################
class SBS():
    #k_features is how many feature we want to keep.
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=self.test_size,
                                 random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        #we keep reduce the features number to K_features
        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            #find the index of the max scores to save to best
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


#######################################################################
# use SBS in Knn for example
# (wine)
#############################



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


    ###################
    # Run KNN
    ###################
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt

    knn = KNeighborsClassifier(n_neighbors=2)

    # selecting features, k_features=1 means we only need 1 feature
    sbs = SBS(knn, k_features=1)
    sbs.fit(X_train_std, y_train)

    ##############################
    # plotting performance of feature subsets
    ###############################
    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.tight_layout()
    # plt.savefig('./sbs.png', dpi=300)
    plt.show()


    #we found , when features number = 5,
    # the Accuracy is get first 100%
    # so, what is these 5 features ?
    k5 = list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])
    #Index(['Alcohol', 'Malic acid', 'Alcalinity of ash', 'Hue', 'Proline'], dtype='object')






    ########################
    # Use ALL FEATURES in KNN
    #########################
    knn.fit(X_train_std, y_train)
    print('Training accuracy:', knn.score(X_train_std, y_train))
    print('Test accuracy:', knn.score(X_test_std, y_test))
    #Training accuracy: 0.983870967742
    #Test accuracy: 0.944444444444
    #Training > Test, a little overfit


    ########################
    # Use 5 FEATURES we selected by SBS to KNN
    #########################
    knn.fit(X_train_std[:, k5], y_train)
    print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
    print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))
    #Training accuracy: 0.959677419355
    #Test accuracy: 0.962962962963
    # testing accuracy gets better



    return 0

main()
