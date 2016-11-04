import os
import struct
import numpy as np
import matplotlib.pyplot as plt

filepath=os.path.dirname(os.path.realpath(__file__))#root, where the apk_integration_test.py file is.
source_folder='source'
#print(filepath)
data_csv_path=filepath+'/'+source_folder

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    '''ref: http://yann.lecun.com/exdb/mnist/ '''
    ''' each hand write is 28x28 = 784, a 1 dim vector'''
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                                % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)

    # check the offical doc to know how to extract the content
    '''
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    '''
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    '''
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        ''' each hand write is 28x28 = 784, a 1 dim vector'''
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def plot_out_ubyte_data(X_train,y_train):

    #######################
    # Visualize the first digit of each class
    ######################

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_all.png', dpi=300)
    plt.show()



    #########################
    # Visualize 25 different versions of "7"
    #########################
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = X_train[y_train == 9][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_7.png', dpi=300)
    plt.show()

    return 0

def read_data_from_by_ubyte():

    X_train, y_train = load_mnist(filepath+'/'+source_folder, kind='train')
    print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

    '''
    Rows: 60000, columns: 784
    '''
    X_test, y_test = load_mnist(filepath+'/'+source_folder, kind='t10k')
    print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

    '''
    Rows: 10000, columns: 784
    '''



    return X_train, y_train, X_test, y_test

def save_byte_data_to_csv(X_train, y_train, X_test, y_test):

    np.savetxt(data_csv_path+'/train_img.csv', X_train, fmt='%i', delimiter=',')
    np.savetxt(data_csv_path+'/train_labels.csv', y_train, fmt='%i', delimiter=',')
    np.savetxt(data_csv_path+'/test_img.csv', X_test, fmt='%i', delimiter=',')
    np.savetxt(data_csv_path+'/test_labels.csv', y_test, fmt='%i', delimiter=',')
    return 0

def read_data_from_csv():
    X_train = np.genfromtxt(data_csv_path+'/train_img.csv', dtype=int, delimiter=',')
    y_train = np.genfromtxt(data_csv_path+'/train_labels.csv', dtype=int, delimiter=',')
    X_test = np.genfromtxt(data_csv_path+'/test_img.csv', dtype=int, delimiter=',')
    y_test = np.genfromtxt(data_csv_path+'/test_labels.csv', dtype=int, delimiter=',')
    return X_train, y_train, X_test, y_test


def plot_to_check_the_converage(nn):

    ###############
    # plot the cost to check converage or not
    ################

    plt.plot(range(len(nn.cost_)), nn.cost_)
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs * 50')
    plt.tight_layout()
    # plt.savefig('./figures/cost.png', dpi=300)
    plt.show()


    # try to use little batch duration to get more smooth pic
    batches = np.array_split(range(len(nn.cost_)), 1000)
    cost_ary = np.array(nn.cost_)
    cost_avgs = [np.mean(cost_ary[i]) for i in batches]
    #plot again, with average cost , will get much more smooth
    plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
    plt.ylim([0, 2000])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.tight_layout()
    #plt.savefig('./figures/cost2.png', dpi=300)
    plt.show()


    return 0


def main():

    #using byte
    X_train, y_train, X_test, y_test=read_data_from_by_ubyte()
    #save_byte_data_to_csv(X_train, y_train, X_test, y_test)

    #using csv, loading time is vary slow, but works
    #X_train, y_train, X_test, y_test=read_data_from_csv()

    #take a look
    #plot_out_ubyte_data(X_train,y_train)

    print("Data ready!")



    #####################
    # Training an artificial neural network
    ####################
    #import from outside we made neuralnet.py
    from neuralnet import NeuralNetMLP
    nn = NeuralNetMLP(n_output=10,
                      n_features=X_train.shape[1],
                      n_hidden=50,
                      l2=0.1,
                      l1=0.0,
                      epochs=1000,
                      eta=0.001,
                      alpha=0.001,
                      decrease_const=0.00001,
                      minibatches=50,
                      random_state=1)
    nn.fit(X_train, y_train, print_progress=True)



    ###############
    # plot the cost to check converage or not
    ################
    #plot_to_check_the_converage(nn)

    ####################
    # do predict
    #####################
    y_train_pred = nn.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))

    y_test_pred = nn.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Testing accuracy: %.2f%%' % (acc * 100))


    #################
    # plot out the misclassify number
    #################
    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab= y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True,)
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    # plt.savefig('./figures/mnist_miscl.png', dpi=300)
    plt.show()



    ##################
    # Debugging neural networks with gradient checking
    # MLPGradientCheck almost the same with NeuralNetMLP
    # only add more gradientcheck fucntion
    ################
    from neuralnet import MLPGradientCheck
    nn_check = MLPGradientCheck(n_output=10,
                                n_features=X_train.shape[1],
                                n_hidden=10,
                                l2=0.0,
                                l1=0.0,
                                epochs=10,
                                eta=0.001,
                                alpha=0.0,
                                decrease_const=0.0,
                                minibatches=1,
                                random_state=1)

    # we do the small sample to quick test
    nn_check.fit(X_train[:5], y_train[:5], print_progress=False)
    '''
    Ok: 2.56712936241e-10
    Ok: 2.94603251069e-10
    Ok: 2.37615620231e-10
    Ok: 2.43469423226e-10
    Ok: 3.37872073158e-10
    Ok: 3.63466384861e-10
    Ok: 2.22472120785e-10
    Ok: 2.33163708438e-10
    Ok: 3.44653686551e-10
    Ok: 2.17161707211e-10
    '''

    return 0


main()
