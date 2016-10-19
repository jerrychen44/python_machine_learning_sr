import pandas as pd
import numpy as np
import os



def read_data():

    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

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


    ###############
    # Bringing features onto the same scale
    # 1.normalization, 2.standardization
    ################
    # 1.normalization with min-max scling
    # betweem 0~1
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    #print(X_train_norm)
    #print(X_test_norm)

    #2. standardization, limit the values
    # in some range.and with the std information
    # mean will =0, std will =1
    from sklearn.preprocessing import StandardScaler

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    print(X_train_std.mean(),X_train_std.std())
    #print(X_test_std)



    #A visual example:
    ex = pd.DataFrame([0, 1, 2 ,3, 4, 5])

    # standardize
    ex[1] = (ex[0] - ex[0].mean()) / ex[0].std()
    # normalize
    ex[2] = (ex[0] - ex[0].min()) / (ex[0].max() - ex[0].min())
    ex.columns = ['input', 'standardized', 'normalized']
    print(ex)
    return 0

main()
