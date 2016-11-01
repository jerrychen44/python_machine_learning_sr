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
    #Solving regression parameters with gradient descent
    ###############################
    # since we though  RM is a good feature
    # we only pick this one to our X
    X = df[['RM']].values
    y = df['MEDV'].values

    #standar
    '''
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    '''

    print("data ready..")






    #################
    # call our model
    #################
    #sklearn_linear_model(X, y)
    sklearn_linear_model_RANSAC(X, y)



    return 0


main()
