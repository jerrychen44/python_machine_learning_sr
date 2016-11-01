import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys
import numpy as np


filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
print(filepath)
data_csv_path=filepath+'/'+source_folder+'/house.csv'



def read_data():
    #ref: https://archive.ics.uci.edu/ml/datasets/Housing
    #df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None, sep='\s+')

    #df.to_csv(data_csv_path,index=0,header=False)


    df = pd.read_csv(data_csv_path, header=None)

    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    '''
    Attributes:
    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over
                     25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds
                     river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                     by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's => y label
    '''
    print(df.head())
    return df


def plot_the_correlation_scatter(df):

    sns.set(style='whitegrid', context='notebook')
    #cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS',
                  'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                  'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    sns.pairplot(df[cols], size=2.5);
    #plt.tight_layout()
    # plt.savefig('./figures/scatter.png', dpi=300)
    plt.show()
    return 0


def plot_the_correlation_matrix_heatmap(df):


    cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    #cols = ['CRIM', 'ZN', 'INDUS', 'CHAS',
    #              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
    #              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 15},
                yticklabels=cols,
                xticklabels=cols)

    #plt.tight_layout()
    # plt.savefig('./figures/corr_mat.png', dpi=300)
    plt.show()

    return 0



def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return



def sklearn_linear_model(X, y):
    #Estimating coefficient of a regression model via scikit-learn
    from sklearn.linear_model import LinearRegression
    slr = LinearRegression()
    slr.fit(X, y)
    y_pred = slr.predict(X)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)
    '''Slope: 9.102
    Intercept: -34.671'''


    #plot output
    lin_regplot(X, y, slr)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    #plt.tight_layout()
    # plt.savefig('./figures/scikit_lr_fit.png', dpi=300)
    plt.show()
    return 0


def sklearn_linear_model_RANSAC(X, y):
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import LinearRegression
    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=50,
                             residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                             residual_threshold=5.0,
                             random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./figures/ransac_fit.png', dpi=300)
    plt.show()


    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)
    '''Slope: 9.621
    Intercept: -37.137'''
    return 0

def plot_residuals(y_train_pred,y_train,y_test_pred,y_test):
    #plot output
    # 1. y_train_pred - y_train means residuals
    # 2. if pred vary well than the y_train_pred - y_train should be = 0
    # you also can found the outlier as the point which much away from the 0. which means the predict is
    # really not much like the y_train.
    # 3. if you can find some pattern in this fig, which means the model missing some feature
    # which can represent the data in your model, so they shows in the result of your predict with some pattern.
    plt.scatter(y_train_pred,  y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    #plt.tight_layout()

    # plt.savefig('./figures/slr_residuals.png', dpi=300)
    plt.show()


    return 0

def main():

    df=read_data()


    ########################
    # Exploratory Data Analysis
    # EDA section
    ########################
    #first look, for Exploratory Data Analysis
    #plot_the_correlation_scatter(df)

    # numberilze the collelation to heatmap
    #plot_the_correlation_matrix_heatmap(df)
    #sns.reset_orig()

    print("We found the LSTAT(-0.74), and RM(0.70) has the larger correlation with our label y MEDV")
    print("We check the scatter again, MEDV and LSTAT not linear correlation, but RM is.")
    print("So, pick RM to be a feature is a good choice.")
    # reset back to the original setting for matplotlib







    ################################
    # Evaluating the performance of linear regression models
    ###############################
    from sklearn.cross_validation import train_test_split

    X = df.iloc[:, :-1].values
    y = df['MEDV'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    print("data ready..")






    #################
    # call our model
    #################
    from sklearn.linear_model import LinearRegression
    slr = LinearRegression()

    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)


    ######################
    # Evaluation method 1: Residuals fig
    ######################
    plot_residuals(y_train_pred,y_train,y_test_pred,y_test)



    ###################
    # Evaluation method 2 : mean_squared_error , MSE and coefficient of determination R^2
    ###################
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    #coefficient of determination
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))


    '''
    MSE train: 19.958, test: 27.196 (testing error > train error ,overfit)
    R^2 train: 0.765, test: 0.673
    '''



    ######################
    # try Using regularized methods for regression to deal overfit
    ######################
    ##############
    # regularized methods LASSO: can also use to feature selection
    ###############
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    print(lasso.coef_)

    plot_residuals(y_train_pred,y_train,y_test_pred,y_test)


    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    #coefficient of determination
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    '''
    [-0.11311792  0.04725111 -0.03992527  0.96478874 -0.          3.72289616
     -0.02143106 -1.23370405  0.20469    -0.0129439  -0.85269025  0.00795847
     -0.52392362]
    MSE train: 20.926, test: 28.876
    R^2 train: 0.753, test: 0.653
    '''



    ##############
    # regularized methods Ridge
    ###############
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)
    print(ridge.coef_)

    plot_residuals(y_train_pred,y_train,y_test_pred,y_test)


    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    #coefficient of determination
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    '''
    [ -1.18308575e-01   4.61259764e-02  -2.08626416e-02   2.45868617e+00
      -8.25958494e+00   3.89748516e+00  -1.79140171e-02  -1.39737175e+00
       2.18432298e-01  -1.16338128e-02  -9.31711410e-01   7.26996266e-03
      -4.94046539e-01]
    MSE train: 20.145, test: 27.762
    R^2 train: 0.762, test: 0.667
    '''
    ##############
    # regularized methods ElasticNet
    ###############
    from sklearn.linear_model import ElasticNet
    elast = ElasticNet(alpha=1.0, l1_ratio =0.5)# is l1_ratio =1 then model ==LASSO
    elast.fit(X_train, y_train)
    y_train_pred = elast.predict(X_train)
    y_test_pred = elast.predict(X_test)
    print(elast.coef_)

    plot_residuals(y_train_pred,y_train,y_test_pred,y_test)


    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    #coefficient of determination
    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))


    '''
    [-0.08344408  0.05179376 -0.01702468  0.         -0.          0.90890973
      0.01218953 -0.83010765  0.23558231 -0.01502425 -0.84881663  0.00687826
     -0.72504946]
    MSE train: 24.381, test: 31.874
    R^2 train: 0.712, test: 0.617
    '''

    return 0


main()
