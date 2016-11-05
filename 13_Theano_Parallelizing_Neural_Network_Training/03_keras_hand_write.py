import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import theano
from theano import tensor as T

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

def main():

    #using byte
    X_train, y_train, X_test, y_test=read_data_from_by_ubyte()


    # Multi-layer Perceptron in Keras

    # In order to run the following code via GPU, you can execute the
    # Python script that was placed in this directory via
    # THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_keras_mlp.py
    theano.config.floatX = 'float32'
    X_train = X_train.astype(theano.config.floatX)
    X_test = X_test.astype(theano.config.floatX)

    #One-hot encoding of the class variable:
    from keras.utils import np_utils
    '''
    ~/.keras/keras.json

    If it isn't there, you can create it.

    The default configuration file looks like this:

    {
        "image_dim_ordering": "tf",
        "epsilon": 1e-07,
        "floatx": "float32",
        "backend": "tensorflow" or "theano" (default is tensorflow)
    }
    '''



    print('First 3 labels: ', y_train[:3])
    '''First 3 labels:  [5 0 4]'''
    y_train_ohe = np_utils.to_categorical(y_train)
    print('\nFirst 3 labels (one-hot):\n', y_train_ohe[:3])
    '''
    First 3 labels (one-hot):
     [[ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]
     '''

    print("Data ready")


    #################
    # Implementing the MLP by keras
    #################
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import SGD

    np.random.seed(1)
    # use Sequential init a new model
    # and use it to Implementing a forward MLP netowrk
    model = Sequential()

    print("!! You can tune the hidden layer number , SGD lr, decay and momentum")
    # add first layer in to model. (input laryer is the first layer)
    # we chose the activation tanh to use.
    model.add(Dense(input_dim=X_train.shape[1],
                    output_dim=50,
                    init='uniform',
                    activation='tanh'))
    # the first layer output dim =50 = second layer input_dim
    model.add(Dense(input_dim=50,
                    output_dim=50,
                    init='uniform',
                    activation='tanh'))

    model.add(Dense(input_dim=50,
                    output_dim=y_train_ohe.shape[1],
                    init='uniform',
                    activation='softmax'))

    # we init a gradient descent to our optimiz function
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
    # we set categorical_crossentropy to set our cost function
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    # train the function
    model.fit(X_train, y_train_ohe,
              nb_epoch=50,
              batch_size=300,
              verbose=1,
              validation_split=0.1,
              show_accuracy=True)
    '''
    Train on 54000 samples, validate on 6000 samples
    Epoch 0
    54000/54000 [==============================] - 1s - loss: 2.2290 - acc: 0.3592 - val_loss: 2.1094 - val_acc: 0.5342
    Epoch 1
    54000/54000 [==============================] - 1s - loss: 1.8850 - acc: 0.5279 - val_loss: 1.6098 - val_acc: 0.5617
    Epoch 2
    54000/54000 [==============================] - 1s - loss: 1.3903 - acc: 0.5884 - val_loss: 1.1666 - val_acc: 0.6707
    Epoch 3
    54000/54000 [==============================] - 1s - loss: 1.0592 - acc: 0.6936 - val_loss: 0.8961 - val_acc: 0.7615
    Epoch 4
    54000/54000 [==============================] - 1s - loss: 0.8528 - acc: 0.7666 - val_loss: 0.7288 - val_acc: 0.8290
    Epoch 5
    54000/54000 [==============================] - 1s - loss: 0.7187 - acc: 0.8191 - val_loss: 0.6122 - val_acc: 0.8603
    Epoch 6
    54000/54000 [==============================] - 1s - loss: 0.6278 - acc: 0.8426 - val_loss: 0.5347 - val_acc: 0.8762
    Epoch 7
    54000/54000 [==============================] - 1s - loss: 0.5592 - acc: 0.8621 - val_loss: 0.4707 - val_acc: 0.8920
    Epoch 8
    54000/54000 [==============================] - 1s - loss: 0.4978 - acc: 0.8751 - val_loss: 0.4288 - val_acc: 0.9033
    Epoch 9
    54000/54000 [==============================] - 1s - loss: 0.4583 - acc: 0.8847 - val_loss: 0.3935 - val_acc: 0.9035
    Epoch 10
    54000/54000 [==============================] - 1s - loss: 0.4213 - acc: 0.8911 - val_loss: 0.3553 - val_acc: 0.9088
    Epoch 11
    54000/54000 [==============================] - 1s - loss: 0.3972 - acc: 0.8955 - val_loss: 0.3405 - val_acc: 0.9083
    Epoch 12
    54000/54000 [==============================] - 1s - loss: 0.3740 - acc: 0.9022 - val_loss: 0.3251 - val_acc: 0.9170
    Epoch 13
    54000/54000 [==============================] - 1s - loss: 0.3611 - acc: 0.9030 - val_loss: 0.3032 - val_acc: 0.9183
    Epoch 14
    54000/54000 [==============================] - 1s - loss: 0.3479 - acc: 0.9064 - val_loss: 0.2972 - val_acc: 0.9248
    Epoch 15
    54000/54000 [==============================] - 1s - loss: 0.3309 - acc: 0.9099 - val_loss: 0.2778 - val_acc: 0.9250
    Epoch 16
    54000/54000 [==============================] - 1s - loss: 0.3264 - acc: 0.9103 - val_loss: 0.2838 - val_acc: 0.9208
    Epoch 17
    54000/54000 [==============================] - 1s - loss: 0.3136 - acc: 0.9136 - val_loss: 0.2689 - val_acc: 0.9223
    Epoch 18
    54000/54000 [==============================] - 1s - loss: 0.3031 - acc: 0.9156 - val_loss: 0.2634 - val_acc: 0.9313
    Epoch 19
    54000/54000 [==============================] - 1s - loss: 0.2988 - acc: 0.9169 - val_loss: 0.2579 - val_acc: 0.9288
    Epoch 20
    54000/54000 [==============================] - 1s - loss: 0.2909 - acc: 0.9180 - val_loss: 0.2494 - val_acc: 0.9310
    Epoch 21
    54000/54000 [==============================] - 1s - loss: 0.2848 - acc: 0.9202 - val_loss: 0.2478 - val_acc: 0.9307
    Epoch 22
    54000/54000 [==============================] - 1s - loss: 0.2804 - acc: 0.9194 - val_loss: 0.2423 - val_acc: 0.9343
    Epoch 23
    54000/54000 [==============================] - 1s - loss: 0.2728 - acc: 0.9235 - val_loss: 0.2387 - val_acc: 0.9327
    Epoch 24
    54000/54000 [==============================] - 1s - loss: 0.2673 - acc: 0.9241 - val_loss: 0.2265 - val_acc: 0.9385
    Epoch 25
    54000/54000 [==============================] - 1s - loss: 0.2611 - acc: 0.9253 - val_loss: 0.2270 - val_acc: 0.9347
    Epoch 26
    54000/54000 [==============================] - 1s - loss: 0.2676 - acc: 0.9225 - val_loss: 0.2210 - val_acc: 0.9367
    Epoch 27
    54000/54000 [==============================] - 1s - loss: 0.2528 - acc: 0.9261 - val_loss: 0.2241 - val_acc: 0.9373
    Epoch 28
    54000/54000 [==============================] - 1s - loss: 0.2511 - acc: 0.9264 - val_loss: 0.2170 - val_acc: 0.9403
    Epoch 29
    54000/54000 [==============================] - 1s - loss: 0.2433 - acc: 0.9293 - val_loss: 0.2165 - val_acc: 0.9412
    Epoch 30
    54000/54000 [==============================] - 1s - loss: 0.2465 - acc: 0.9279 - val_loss: 0.2135 - val_acc: 0.9367
    Epoch 31
    54000/54000 [==============================] - 1s - loss: 0.2383 - acc: 0.9306 - val_loss: 0.2138 - val_acc: 0.9427
    Epoch 32
    54000/54000 [==============================] - 1s - loss: 0.2349 - acc: 0.9310 - val_loss: 0.2066 - val_acc: 0.9423
    Epoch 33
    54000/54000 [==============================] - 1s - loss: 0.2301 - acc: 0.9334 - val_loss: 0.2054 - val_acc: 0.9440
    Epoch 34
    54000/54000 [==============================] - 1s - loss: 0.2371 - acc: 0.9317 - val_loss: 0.1991 - val_acc: 0.9480
    Epoch 35
    54000/54000 [==============================] - 1s - loss: 0.2256 - acc: 0.9352 - val_loss: 0.1982 - val_acc: 0.9450
    Epoch 36
    54000/54000 [==============================] - 1s - loss: 0.2313 - acc: 0.9323 - val_loss: 0.2092 - val_acc: 0.9403
    Epoch 37
    54000/54000 [==============================] - 1s - loss: 0.2230 - acc: 0.9341 - val_loss: 0.1993 - val_acc: 0.9445
    Epoch 38
    54000/54000 [==============================] - 1s - loss: 0.2261 - acc: 0.9336 - val_loss: 0.1891 - val_acc: 0.9463
    Epoch 39
    54000/54000 [==============================] - 1s - loss: 0.2166 - acc: 0.9369 - val_loss: 0.1943 - val_acc: 0.9452
    Epoch 40
    54000/54000 [==============================] - 1s - loss: 0.2128 - acc: 0.9370 - val_loss: 0.1952 - val_acc: 0.9435
    Epoch 41
    54000/54000 [==============================] - 1s - loss: 0.2200 - acc: 0.9351 - val_loss: 0.1918 - val_acc: 0.9468
    Epoch 42
    54000/54000 [==============================] - 2s - loss: 0.2107 - acc: 0.9383 - val_loss: 0.1831 - val_acc: 0.9483
    Epoch 43
    54000/54000 [==============================] - 1s - loss: 0.2020 - acc: 0.9411 - val_loss: 0.1906 - val_acc: 0.9443
    Epoch 44
    54000/54000 [==============================] - 1s - loss: 0.2082 - acc: 0.9388 - val_loss: 0.1838 - val_acc: 0.9457
    Epoch 45
    54000/54000 [==============================] - 1s - loss: 0.2048 - acc: 0.9402 - val_loss: 0.1817 - val_acc: 0.9488
    Epoch 46
    54000/54000 [==============================] - 1s - loss: 0.2012 - acc: 0.9417 - val_loss: 0.1876 - val_acc: 0.9480
    Epoch 47
    54000/54000 [==============================] - 1s - loss: 0.1996 - acc: 0.9423 - val_loss: 0.1792 - val_acc: 0.9502
    Epoch 48
    54000/54000 [==============================] - 1s - loss: 0.1921 - acc: 0.9430 - val_loss: 0.1791 - val_acc: 0.9505
    Epoch 49
    54000/54000 [==============================] - 1s - loss: 0.1907 - acc: 0.9432 - val_loss: 0.1749 - val_acc: 0.9482
    Using Theano backend.
    '''

    # use the model to predict
    y_train_pred = model.predict_classes(X_train, verbose=0)
    print('First 3 predictions: ', y_train_pred[:3])
    ''' First 3 predictions:  [5 0 4] '''
    #list out the acc for training set and testing set .
    train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (train_acc * 100))
    ''' Training accuracy: 94.17% '''

    y_test_pred = model.predict_classes(X_test, verbose=0)
    test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (test_acc * 100))
    ''' Test accuracy: 93.67% '''
    return 0

main()
