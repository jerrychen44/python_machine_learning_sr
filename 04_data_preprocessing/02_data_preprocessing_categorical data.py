import pandas as pd
import numpy as np

df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']




def main():
    ###################
    #Step1, get data
    #################
    print(df)

    #################################
    #Step2: Handling categorical data : change the order categorical to nmuber
    ###################################
    #define the mapping table by yourself for order feature
    size_mapping = {
           'XL': 3,
           'L': 2,
           'M': 1}

    df['size'] = df['size'].map(size_mapping)
    print(df)

    #if we want to revirse back
    '''
    inv_size_mapping = {v: k for k, v in size_mapping.items()}
    df['size'] =df['size'].map(inv_size_mapping)
    print(df)
    '''


    #############################
    # Step3: Encoding Class labels: change label to number (no order)
    ##############################

    ############
    #  Step3: Mathod A, trandition way
    ###########
    '''
    class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
    print(class_mapping)
    df['classlabel'] = df['classlabel'].map(class_mapping)
    print(df)
    '''
    #if we want to revirse back
    '''
    inv_class_mapping = {v: k for k, v in class_mapping.items()}
    df['classlabel'] = df['classlabel'].map(inv_class_mapping)
    print(df)
    '''

    ############
    #  Step3: Method B, skleanr
    ###########
    from sklearn.preprocessing import LabelEncoder

    class_le = LabelEncoder()
    df['classlabel'] = class_le.fit_transform(df['classlabel'].values)
    print(df)
    #if we want to revirse back
    '''
    df['classlabel'] = class_le.inverse_transform(df['classlabel'].values)
    print(df)
    '''


    ###################
    #Step4, Encoding the nominal features with onehot enconding
    #################
    ###### Method A , trandition way #########
    '''
    #need to change naming to nmber first
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
    print(X)

    #do onehot enconding to avoid the order side effect
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(categorical_features=[0])
    result = ohe.fit_transform(X).toarray()
    print(result)
    '''


    ###### Method B , fast way #########
    #get_dummies will automatic onehot enconding for all string value(feature)
    result=pd.get_dummies(df[['price', 'color', 'size']])
    print(result)

    return 0


main()
