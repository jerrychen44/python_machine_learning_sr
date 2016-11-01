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

##########################
# Implementing at a simple regression model - Ordinary least squares
########################
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return





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

    print("We found the LSTAT(-0.74), and RM(0.70) has the larger correlation with our label y MEDV")
    print("We check the scatter again, MEDV and LSTAT not linear correlation, but RM is.")
    print("So, pick RM to be a feature is a good choice.")
    # reset back to the original setting for matplotlib
    sns.reset_orig()






    ################################
    #Solving regression parameters with gradient descent
    ###############################
    # since we though  RM is a good feature
    # we only pick this one to our X
    X = df[['RM']].values
    y = df['MEDV'].values

    #standar
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)


    #################
    # call our model
    #################
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)

    #plot the result
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.tight_layout()
    #plt.savefig('./figures/cost.png', dpi=300)
    plt.show()
    print("We found the converage at the 5 round.")


    #######################
    # We put the original data back and the linear model together.
    ########################
    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.tight_layout()
    # plt.savefig('./figures/gradient_fit.png', dpi=300)
    plt.show()
    print('Slope: %.3f' % lr.w_[1])
    print('Intercept: %.3f' % lr.w_[0])# should be 0 is because we standardized it before.
    '''
    Slope: 0.695
    Intercept: -0.000
    '''

    #inver back to the us doller for label y
    # assume we want to know the price with 5 rooms
    # it predict to 10.840 x $1000 = $10,840 us doller
    num_rooms_std = sc_x.transform([5.0])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))
    '''Price in $1000's: 10.840'''




    return 0


main()
