import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt


#############################
# Logistic function recap
##############################
def net_input(X, w):
    z = X.dot(w)
    return z

def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_activation(X, w):
    z = net_input(X, w)
    return logistic(z)

def logistic_activation_function():
    # note that first element (X[0] = 1) to denote bias unit

    X = np.array([[1, 1.4, 1.5]])
    w = np.array([0.0, 0.2, 0.4])


    print('P(y=1|x) = %.3f' % logistic_activation(X, w)[0])
    '''P(y=1|x) = 0.707'''


    #Now, imagine a MLP perceptron with 3 hidden units + 1
    #bias unit in the hidden unit. The output layer consists of 3 output units.

    # W : array, shape = [n_output_units, n_hidden_units+1]
    #          Weight matrix for hidden layer -> output layer.
    # note that first column (A[:][0] = 1) are the bias units
    W = np.array([[1.1, 1.2, 1.3, 0.5],
                  [0.1, 0.2, 0.4, 0.1],
                  [0.2, 0.5, 2.1, 1.9]])

    # A : array, shape = [n_hidden+1, n_samples]
    #          Activation of hidden layer.
    # note that first element (A[0][0] = 1) is for the bias units

    A = np.array([[1.0],
                  [0.1],
                  [0.3],
                  [0.7]])

    # Z : array, shape = [n_output_units, n_samples]
    #          Net input of output layer.

    Z = W.dot(A)
    y_probas = logistic(Z)
    print('Probabilities:\n', y_probas)
    '''
     Probabilities:
     [[ 0.87653295]
     [ 0.57688526]
     [ 0.90114393]]
    '''
    ###########
    # chose the hieghtest probabilites
    #############
    y_class = np.argmax(Z, axis=0)
    print('predicted class label: %d' % y_class[0])
    ''' predicted class label: 2 '''
    return 0


#########################
# softmax : like a normalized sigmoid function
#######################
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

def softmax_activation(X, w):
    z = net_input(X, w)
    return sigmoid(z)




# class-probalities in multi-class
def softmax_activation_function():

    # W : array, shape = [n_output_units, n_hidden_units+1]
    #          Weight matrix for hidden layer -> output layer.
    # note that first column (A[:][0] = 1) are the bias units
    W = np.array([[1.1, 1.2, 1.3, 0.5],
                  [0.1, 0.2, 0.4, 0.1],
                  [0.2, 0.5, 2.1, 1.9]])

    # A : array, shape = [n_hidden+1, n_samples]
    #          Activation of hidden layer.
    # note that first element (A[0][0] = 1) is for the bias units

    A = np.array([[1.0],
                  [0.1],
                  [0.3],
                  [0.7]])

    # Z : array, shape = [n_output_units, n_samples]
    #          Net input of output layer.

    Z = W.dot(A)

    y_probas = softmax(Z)
    print('Probabilities:\n', y_probas)
    '''
    Probabilities:
     [[ 0.40386493]
     [ 0.07756222]
     [ 0.51857284]]
     '''

    print(y_probas.sum())
    '''1.0'''


    y_class = np.argmax(Z, axis=0)
    print(y_class)
    '''[2]'''
    return 0

######################
# hyperbolic tangent : a scaled logistic function  -1~1
######################
def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)


def plot_to_compare_logistic_tanh():
    z = np.arange(-5, 5, 0.005)
    log_act = logistic(z)
    tanh_act = tanh(z)

    # alternatives: the same with above
    #from scipy.special import expit
    #log_act = expit(z)
    #tanh_act = np.tanh(z)

    plt.ylim([-1.5, 1.5])
    plt.xlabel('net input $z$')
    plt.ylabel('activation $\phi(z)$')
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle='--')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(-1, color='black', linestyle='--')

    plt.plot(z, tanh_act,
             linewidth=2,
             color='black',
             label='tanh')
    plt.plot(z, log_act,
             linewidth=2,
             color='lightgreen',
             label='logistic')

    plt.legend(loc='lower right')
    plt.tight_layout()
    # plt.savefig('./figures/activation.png', dpi=300)
    plt.show()
    return 0

def hyperbolic_tangent_activation_function():

    plot_to_compare_logistic_tanh()


    return 0
def main():
    # Choosing activation functions for feedforward neural networks

    #####################
    # activation function 1: Sigmoid fun
    #######################
    # Logistic function recap
    logistic_activation_function()

    #####################
    # activation function 2: softmax fun
    #######################
    # Estimating probabilities in multi-class
    # classification via the softmax function
    softmax_activation_function()


    #####################
    # activation function 3:  hyperbolic tangent fun
    #######################
    hyperbolic_tangent_activation_function()
    return 0

main()
