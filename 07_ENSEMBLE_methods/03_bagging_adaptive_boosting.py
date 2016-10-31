import pandas as pd

def read_data():
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

    # drop 1 class, in this case, we only consider it
    # to binary classify problem
    df_wine = df_wine[df_wine['Class label'] != 1]



    return df_wine

def bagging(X,y,X_train,y_train,X_test,y_test):

    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=None)

    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500,
                            max_samples=1.0,
                            max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=1,
                            random_state=1)

    #quick check the accuracy between tree and bag
    from sklearn.metrics import accuracy_score

    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)

    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print('Bagging train/test accuracies %.3f/%.3f'
          % (bag_train, bag_test))
    '''
    Decision tree train/test accuracies 1.000/0.854
    Bagging train/test accuracies 1.000/0.896
    '''
    #take a look
    import numpy as np
    import matplotlib.pyplot as plt

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(nrows=1, ncols=2,
                            sharex='col',
                            sharey='row',
                            figsize=(8, 3))


    for idx, clf, tt in zip([0, 1],
                            [tree, bag],
                            ['Decision Tree', 'Bagging']):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train==0, 0],
                           X_train[y_train==0, 1],
                           c='blue', marker='^')

        axarr[idx].scatter(X_train[y_train==1, 0],
                           X_train[y_train==1, 1],
                           c='red', marker='o')

        axarr[idx].set_title(tt)

    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(10.2, -1.2,
             s='Hue',
             ha='center', va='center', fontsize=12)

    plt.tight_layout()
    #plt.savefig('./figures/bagging_region.png',
    #            dpi=300,
    #            bbox_inches='tight')
    plt.show()


    return 0


def adaptive_boosting(X,y,X_train,y_train,X_test,y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=1)

    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500,
                             learning_rate=0.1,
                             random_state=0)


    from sklearn.metrics import accuracy_score
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)

    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (ada_train, ada_test))

    '''
    Decision tree train/test accuracies 0.845/0.854
    AdaBoost train/test accuracies 1.000/0.875
    '''
    #take a look
    import numpy as np
    import matplotlib.pyplot as plt
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(8, 3))


    for idx, clf, tt in zip([0, 1],
                            [tree, ada],
                            ['Decision Tree', 'AdaBoost']):
        clf.fit(X_train, y_train)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train==0, 0],
                           X_train[y_train==0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train==1, 0],
                           X_train[y_train==1, 1],
                           c='red', marker='o')
        axarr[idx].set_title(tt)

    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(10.2, -1.2,
             s='Hue',
             ha='center', va='center', fontsize=12)

    plt.tight_layout()
    #plt.savefig('./figures/adaboost_region.png',
    #           dpi=300,
    #           bbox_inches='tight')
    plt.show()


    return 0


def main():
    df_wine=read_data()

    # we only check two feature for this example
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'Hue']].values


    from sklearn.preprocessing import LabelEncoder
    from sklearn.cross_validation import train_test_split

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test =\
                train_test_split(X, y,
                                 test_size=0.40,
                                 random_state=1)

    # data ready here.

    #Bagging -- Building an ensemble of classifiers from bootstrap samples¶
    #bagging(X,y,X_train,y_train,X_test,y_test)

    #Leveraging of weak learners via adaptive boosting
    adaptive_boosting(X,y,X_train,y_train,X_test,y_test)


    return 0
main()
