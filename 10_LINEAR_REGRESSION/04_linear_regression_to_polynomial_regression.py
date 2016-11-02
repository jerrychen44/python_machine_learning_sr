import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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

def quick_test():
    X = np.array([258.0, 270.0, 294.0,
              320.0, 342.0, 368.0,
              396.0, 446.0, 480.0, 586.0])[:, np.newaxis]

    y = np.array([236.4, 234.4, 252.8,
                  298.6, 314.2, 342.2,
                  360.8, 368.0, 391.2,
                  390.8])




    # fit linear features
    lr = LinearRegression()
    lr.fit(X, y)
    X_fit = np.arange(250,600,10)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)


    # fit quadratic features
    pr = LinearRegression()

    quadratic = PolynomialFeatures(degree=2)
    #we first transfer 1 dim X to quadratic X
    X_quad = quadratic.fit_transform(X)

    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))



    # plot results
    plt.scatter(X, y, label='training points')
    plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
    plt.plot(X_fit, y_quad_fit, label='quadratic fit')
    plt.legend(loc='upper left')

    plt.tight_layout()
    # plt.savefig('./figures/poly_example.png', dpi=300)
    plt.show()


    # get predict output
    y_lin_pred = lr.predict(X)
    y_quad_pred = pr.predict(X_quad)



    print('Training MSE linear: %.3f, quadratic: %.3f' % (
            mean_squared_error(y, y_lin_pred),
            mean_squared_error(y, y_quad_pred)))
    print('Training  R^2 linear: %.3f, quadratic: %.3f' % (
            r2_score(y, y_lin_pred),
            r2_score(y, y_quad_pred)))
    '''
    Training MSE linear: 569.780, quadratic: 61.330
    Training  R^2 linear: 0.832, quadratic: 0.982
    '''
    return 0



def modeling_polynomial_house(df):
    # only check the LSTAT with MEDV
    X = df[['LSTAT']].values
    y = df['MEDV'].values

    regr = LinearRegression()

    # create quadratic features
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # fit features
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]

    regr = regr.fit(X, y)# original linear X
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    regr = regr.fit(X_quad, y)# original quad X
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    regr = regr.fit(X_cubic, y)# original cubic X
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))


    # plot results
    plt.scatter(X, y, label='training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2,
             linestyle=':')

    plt.plot(X_fit, y_quad_fit,
             label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
             color='red',
             lw=2,
             linestyle='-')

    plt.plot(X_fit, y_cubic_fit,
             label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
             color='green',
             lw=2,
             linestyle='--')

    plt.xlabel('% lower status of the population [LSTAT]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper right')

    #plt.tight_layout()
    # plt.savefig('./figures/polyhouse_example.png', dpi=300)
    plt.show()

    return 0


def plot_the_correlation_scatter(df):

    sns.set(style='whitegrid', context='notebook')
    #cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
    cols = ['LSTAT', 'MEDV']

    sns.pairplot(df[cols], size=2.5);
    #plt.tight_layout()
    # plt.savefig('./figures/scatter.png', dpi=300)
    plt.show()
    sns.reset_orig()
    return 0
def transforming_non_linear_dataset_tofix_linear(df):

    #before transforming, take a look.
    #plot_the_correlation_scatter(df)

    X = df[['LSTAT']].values
    y = df['MEDV'].values
    print(X,type(X))
    print(y,type(y))
    # transform features
    X_log = np.log(X)
    #X_log=X
    y_sqrt = np.sqrt(y)
    #y_sqrt=y
    # fit features
    X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]

    regr = LinearRegression()
    regr = regr.fit(X_log, y_sqrt)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

    # plot results
    plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')

    plt.plot(X_fit, y_lin_fit,
             label='linear (d=1), $R^2=%.2f$' % linear_r2,
             color='blue',
             lw=2)

    plt.xlabel('log(% lower status of the population [LSTAT])')
    plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
    plt.legend(loc='lower left')

    #plt.tight_layout()
    # plt.savefig('./figures/transform_example.png', dpi=300)
    plt.show()
    ''' R=0.69 '''

    return 0


def example_x_log_and_y_sqrt():

    ################
    # exponential to log
    ###############
    X = np.linspace(-1, 2, 50)# for axis
    y = np.exp(-X)# first fake data with exponential
    plt.scatter(X, y, label='original X with exponential', color='green')

    y_log=np.log(y)# change exponential data to log
    #print(X,type(X))
    plt.scatter(X, y_log, label='log(X)', color='blue')
    plt.show()

    ################
    # exponential to sqrt
    ###############
    X = np.linspace(-1, 2, 50)# for axis
    y = np.exp(-X)# first fake data with exponential
    plt.scatter(X, y, label='original X with exponential', color='green')

    y_sqrt=np.sqrt(y)# change exponential data to log
    #print(X,type(X))
    plt.scatter(X, y_sqrt, label='sqrt(X)', color='blue')
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


    print("data ready..")






    #################
    # call our model
    #################
    #quick_test()
    print("The second high correlation feature is LASTAT, let check it out.")
    modeling_polynomial_house(df)

    print("not everytime you need to use polynomial regression")
    print("you can transfore X,y to other space to still use linear to fit and also good.")
    transforming_non_linear_dataset_tofix_linear(df)


    example_x_log_and_y_sqrt()

    return 0


main()
