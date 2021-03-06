import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
def read_data():

    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    #ref detail : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.names


    #read from csv
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/breast_cancer.csv'
    df = pd.read_csv(data_csv_path, header=None)


    #save csv
    '''
    filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
    source_folder='source'
    data_csv_path=filepath+'/'+source_folder+'/breast_cancer.csv'
    df.to_csv(data_csv_path,index=0,header=False)
    '''

    #print('Class labels', np.unique(df['Class label']))
    print(df.head())
    print(df.shape)
    return df




def main():
    df=read_data()

    ################
    # data preprocessing
    ################
    from sklearn.preprocessing import LabelEncoder
    #assign 30 features set to X.
    X = df.loc[:, 2:].values
    #assign [1] to label y
    y = df.loc[:, 1].values
    #print(y)

    #transform y label M,B to 1,0
    le = LabelEncoder()
    y = le.fit_transform(y)
    le.transform(['M', 'B'])
    #print(y)

    ############
    # separate train, test set
    #############
    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


    print("Data ready")



    ###################
    # use pipeline
    ###################
    #Combining transformers and estimators in a pipeline

    from sklearn.preprocessing import StandardScaler
    #model
    from sklearn.svm import SVC #SVM model
    from sklearn.pipeline import Pipeline

    pipe_svc = Pipeline([('scl', StandardScaler()),
                ('clf', SVC(random_state=1))])


    ##################
    # grid_search for tuning parameter : brute-force all possbility.
    ###################
    from sklearn.grid_search import GridSearchCV

    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    param_grid = [{'clf__C': param_range,# for linear svm to use
                   'clf__kernel': ['linear']},

                     {'clf__C': param_range,# for rbf svm to use
                      'clf__gamma': param_range,
                      'clf__kernel': ['rbf']}]


    ###############
    # nested cross-validation : for SVM
    ################
    from sklearn.cross_validation import cross_val_score
    gs = GridSearchCV(estimator=pipe_svc,
                                param_grid=param_grid,
                                scoring='accuracy',
                                cv=5)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    '''CV accuracy: 0.978 +/- 0.012'''


    ###############
    # nested cross-validation : for DecisionTree
    ################
    from sklearn.tree import DecisionTreeClassifier
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                                param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                                scoring='accuracy',
                                cv=5)
    scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    '''CV accuracy: 0.908 +/- 0.045'''

    # svm 0.97 > dtree 0.90, if some new data came from the same data group, svm
    # will be better.





    #We can make ourown scorer, default pos_label =1,
    # we can change pos_label = 0
    from sklearn.metrics import make_scorer, f1_score

    # I want to see the f1_score as score
    scorer = make_scorer(f1_score, pos_label=0)

    c_gamma_range = [0.01, 0.1, 1.0, 10.0]

    param_grid = [{'clf__C': c_gamma_range,
                   'clf__kernel': ['linear']},
                     {'clf__C': c_gamma_range,
                      'clf__gamma': c_gamma_range,
                      'clf__kernel': ['rbf'],}]

    gs = GridSearchCV(estimator=pipe_svc,
                                    param_grid=param_grid,
                                    scoring=scorer,
                                    cv=10,
                                    n_jobs=-1)
    gs = gs.fit(X_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    '''
    0.982798668208
    {'clf__C': 0.1, 'clf__kernel': 'linear'}
    '''



    return 0

main()
