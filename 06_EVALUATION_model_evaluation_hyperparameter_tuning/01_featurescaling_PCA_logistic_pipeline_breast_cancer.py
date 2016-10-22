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


    return 0

main()
