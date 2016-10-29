import pandas as pd
import os


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
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # StandardScaler for feature scaling
    # PCA for dimensional reduction (30 -> 2)
    # LogisticRegression for prdictive model (our learning algorithm)
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=2)),('clf', LogisticRegression(random_state=1))])

    pipe_lr.fit(X_train, y_train)
    print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
    'Test Accuracy: 0.947'
    y_pred = pipe_lr.predict(X_test)



    ##################
    # Use kfold to do above pipeline again to compare.
    # but first, we calculate score by ourself
    ##################
    import numpy as np
    from sklearn.cross_validation import StratifiedKFold
    #kfold is a index set.
    kfold = StratifiedKFold(y=y_train,
                            n_folds=10,
                            random_state=1)
    #print(kfold)
    scores = []
    for k, (train, test) in enumerate(kfold):
        #print(train,test)
        pipe_lr.fit(X_train[train], y_train[train])
        score = pipe_lr.score(X_train[test], y_train[test])
        scores.append(score)
        print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

    print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    '''
    Fold: 1, Class dist.: [256 153], Acc: 0.891
    Fold: 2, Class dist.: [256 153], Acc: 0.978
    Fold: 3, Class dist.: [256 153], Acc: 0.978
    Fold: 4, Class dist.: [256 153], Acc: 0.913
    Fold: 5, Class dist.: [256 153], Acc: 0.935
    Fold: 6, Class dist.: [257 153], Acc: 0.978
    Fold: 7, Class dist.: [257 153], Acc: 0.933
    Fold: 8, Class dist.: [257 153], Acc: 0.956
    Fold: 9, Class dist.: [257 153], Acc: 0.978
    Fold: 10, Class dist.: [257 153], Acc: 0.956

    CV accuracy: 0.950 +/- 0.029
    '''

    ########################
    # scorer of kfolder by sklearn (much easy way)
    #########################
    from sklearn.cross_validation import cross_val_score

    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10, #10 folder
                             n_jobs=1)# n_jobs name use how many cpu to run. -1 = use all cpu you have.
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    '''
    # the result is the same as we calculate in above step
    CV accuracy scores: [ 0.89130435  0.97826087  0.97826087  0.91304348  0.93478261  0.97777778
      0.93333333  0.95555556  0.97777778  0.95555556]
    CV accuracy: 0.950 +/- 0.029
    '''
    return 0

main()
