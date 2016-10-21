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

    #########################
    # Assessing Feature Importances with Random Forests
    #########################
    from sklearn.ensemble import RandomForestClassifier

    feat_labels = df_wine.columns[1:]

    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)

    forest.fit(X_train, y_train)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]
    #print out the sorting list
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[f],
                                importances[indices[f]]))
    '''
     #importances sorted list
     1) Alcohol                        0.182483
     2) Malic acid                     0.158610
     3) Ash                            0.150948
     4) Alcalinity of ash              0.131987
     5) Magnesium                      0.106589
     6) Total phenols                  0.078243
     7) Flavanoids                     0.060718
     8) Nonflavanoid phenols           0.032033
     9) Proanthocyanins                0.025400
    10) Color intensity                0.022351
    11) Hue                            0.022078
    12) OD280/OD315 of diluted wines   0.014645
    13) Proline                        0.013916
    '''
    #plot out
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')

    plt.xticks(range(X_train.shape[1]),
               feat_labels, rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    # plt.savefig('./figures/random_forest.png', dpi=300)
    plt.show()

    ################
    # give a example if you want to
    # find the most 3 features, set threshold=0.15
    # we will know why is 0.15 later
    ################
    X_selected = forest.transform(X_train, threshold=0.15)
    print(X_selected.shape)

    return 0

main()
