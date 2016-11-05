import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt



def playaround():
    # initialize
    x1 = T.scalar()
    print(x1)
    w1 = T.scalar()
    print(w1)
    w0 = T.scalar()
    print(w0)
    z1 = w1 * x1 + w0
    print(z1)


    # compile
    net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)

    # execute
    result=net_input(2.0, 1.0, 0.5)
    print(result,type(result))
    '''2.5 <class 'numpy.ndarray'> '''
    return 0

def check_and_config_theano_config():


    ####################
    # floating type
    #####################
    print(theano.config.floatX)

    '''
    float64
    '''
    # since GPU use float32
    # Note that float32 is recommended for GPUs;
    # float64 on GPUs is currently still relatively slow.

    # To change the float type globally, execute
    # export THEANO_FLAGS=floatX=float32
    # in your bash shell. Or execute Python script as
    # THEANO_FLAGS=floatX=float32 python your_script.py

    # set to use float32
    theano.config.floatX = 'float32'
    print(theano.config.floatX)
    ''' float32 '''


    ########################
    # execute on....
    #########################
    print(theano.config.device)
    ''' cpu'''

    # You can run a Python script on CPU via:
    # THEANO_FLAGS=device=cpu,floatX=float64 python your_script.py
    # or GPU via
    # THEANO_FLAGS=device=gpu,floatX=float32 python your_script.py
    # It may also be convenient to create a .theanorc file in your home directory to make those configurations permanent. For example, to always use float32, execute
    # echo -e "\n[global]\nfloatX=float32\n" >> ~/.theanorc
    # Or, create a .theanorc file manually with the following contents
    # [global]
    # floatX = float32
    # device = gpu


    return 0

def theano_array_structure_from_list_narray():


    # initialize
    #use fmatrix to create a new tensorvariable
    x = T.fmatrix(name='x')
    print(x,x.type)
    x_sum = T.sum(x, axis=0)# axis 0 is Column

    # compile
    calc_sum = theano.function(inputs=[x], outputs=x_sum)

    # execute (Python list)
    # teano can accept list
    ary = [[1, 2, 3], [1, 2, 3]]
    print(ary)
    ''' [[1, 2, 3], [1, 2, 3]] '''
    print('Column sum:', calc_sum(ary))
    '''Column sum: [ 2.  4.  6.]'''


    # execute (in NumPy array)
    ary = np.array([[1, 2, 3], [1, 2, 3]], dtype=theano.config.floatX)
    print(ary)
    '''
    [[ 1.  2.  3.]
     [ 1.  2.  3.]]
     '''
    # teano can accept teh array from numpy
    print('Column sum:', calc_sum(ary))
    '''Column sum: [ 2.  4.  6.]'''
    return 0


def main_memory_shared_example():
    # Updating shared arrays. More info about memory management in
    #Theano can be found here: http://deeplearning.net/software/theano/tutorial/aliasing.html

    # initialize
    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                                 dtype=theano.config.floatX))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]

    # compile
    net_input = theano.function(inputs=[x],
                                updates=update,
                                outputs=z)

    # execute
    data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
    for i in range(5):
        print('z%d:' % i, net_input(data))

    '''
    z0: [[ 0.]]
    z1: [[ 6.]]
    z2: [[ 12.]]
    z3: [[ 18.]]
    z4: [[ 24.]]
    '''

    ###########################################
    # We can use the givens variable to insert values into the graph before
    # compiling it. Using this approach we can reduce the number of transfers
    # from RAM (via CPUs) to GPUs to speed up learning with shared variables.
    # If we use inputs, a datasets is transferred from the CPU to the GPU
    # multiple times, for example, if we iterate over a dataset multiple times
    # (epochs) during gradient descent. Via givens, we can keep the dataset
    # on the GPU if it fits (e.g., a mini-batch).
    ###########################################

    # initialize
    data = np.array([[1, 2, 3]],
                    dtype=theano.config.floatX)
    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray([[0.0, 0.0, 0.0]],
                                 dtype=theano.config.floatX))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]

    # compile
    net_input = theano.function(inputs=[],
                                updates=update,
                                givens={x: data},
                                outputs=z)

    # execute
    for i in range(5):
        print('z:', net_input())
    '''
    z: [[ 0.]]
    z: [[ 6.]]
    z: [[ 12.]]
    z: [[ 18.]]
    z: [[ 24.]]
    '''



    return 0


###################
# Implementing the training function.
###################
def train_linreg(X_train, y_train, eta, epochs):

    costs = []
    # Initialize arrays
    eta0 = T.fscalar('eta0')
    y = T.fvector(name='y')
    X = T.fmatrix(name='X')
    w = theano.shared(np.zeros(
                        shape=(X_train.shape[1] + 1),
                        dtype=theano.config.floatX),
                      name='w')

    # calculate cost
    net_input = T.dot(X, w[1:]) + w[0]
    errors = y - net_input
    cost = T.sum(T.pow(errors, 2))

    # perform gradient update
    gradient = T.grad(cost, wrt=w)
    update = [(w, w - eta0 * gradient)]

    # compile model
    train = theano.function(inputs=[eta0],
                            outputs=cost,
                            updates=update,
                            givens={X: X_train,
                                    y: y_train,})

    for _ in range(epochs):
        costs.append(train(eta))

    return costs, w


#####################
# Making predictions.
#####################

def predict_linreg(X, w):
    Xt = T.matrix(name='X')
    net_input = T.dot(Xt, w[1:]) + w[0]
    predict = theano.function(inputs=[Xt], givens={w: w}, outputs=net_input)
    return predict(X)


def main():
    #Building, compiling, and running expressions with Theano


    check_and_config_theano_config()



    ######################
    # example secction
    #######################
    #playaround()

    # Working with array structures
    #theano_array_structure_from_list_narray()

    # memory_management
    #main_memory_shared_example()


    ##################
    # Wrapping things up: A linear regression example
    ###################
    #Creating some training data.
    X_train = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0],
                          [5.0], [6.0], [7.0], [8.0], [9.0]],
                         dtype=theano.config.floatX)

    y_train = np.asarray([1.0, 1.3, 3.1, 2.0, 5.0,
                          6.3, 6.6, 7.4, 8.0, 9.0],
                         dtype=theano.config.floatX)


    # training function.
    costs, w = train_linreg(X_train, y_train, eta=0.001, epochs=10)

    #Plotting the sum of squared errors cost vs epochs.
    plt.plot(range(1, len(costs)+1), costs)

    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.tight_layout()
    # plt.savefig('./figures/cost_convergence.png', dpi=300)
    plt.show()


    ####################
    # Making predictions.
    #################
    plt.scatter(X_train, y_train, marker='s', s=50)
    plt.plot(range(X_train.shape[0]),
             predict_linreg(X_train, w),
             color='gray',
             marker='o',
             markersize=4,
             linewidth=3)

    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    # plt.savefig('./figures/linreg.png', dpi=300)
    plt.show()



    return 0


main()
