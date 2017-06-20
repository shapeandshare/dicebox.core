"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import logging
import numpy
import dicebox_config
import filesystem_connecter as fsc

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

# Checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [early_stopper, checkpoint]

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist_filesystem():
    nb_classes = 10
    batch_size = 1000
    input_shape = (784,)

    noise = 1.0
    train_batch_size = 60000
    test_batch_size = 10000

    network_input_index = fsc.get_data_set(config.DATA_DIRECTORY)
    category_map = fsc.get_data_set_categories(network_input_index)

    train_image_data, train_image_labels = fsc.get_batch(config.DATA_DIRECTORY, network_input_index,
                                                       train_batch_size, category_map, noise=noise)
    train_image_data = numpy.array(train_image_data)
    train_image_data = train_image_data.astype('float32')
    train_image_data /= 255
    train_image_labels = numpy.array(train_image_labels)


    test_image_data, test_image_labels = fsc.get_batch(config.DATA_DIRECTORY, network_input_index,
                                             test_batch_size, category_map, noise=noise)
    test_image_data = numpy.array(test_image_data)
    test_image_data = test_image_data.astype('float32')
    test_image_data /= 255
    test_image_labels = numpy.array(test_image_labels)

    logging.info("nb_classes: (%i)" % nb_classes)
    logging.info("batch_size: (%i)" % batch_size)
    logging.info("input_shape: (%s)" % input_shape)
    logging.info("network_input_index: (%s)" % network_input_index)
    logging.info("category_map: (%s)" % category_map)

    #
    #return (nb_classes, batch_size, input_shape, x_test, y_test)
    #return (nb_classes, batch_size, input_shape, image_data, image_labels)
    x_train = train_image_data
    x_test = test_image_data
    y_train = train_image_labels
    y_test = test_image_labels
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_dicebox_filesystem():
    nb_classes = 5
    batch_size = 1000
    input_shape = (3000,)

    noise = 1.0
    train_batch_size = 5000
    test_batch_size = 1000

    network_input_index = fsc.get_data_set(dicebox_config.DATA_DIRECTORY)
    category_map = fsc.get_data_set_categories(network_input_index)

    train_image_data, train_image_labels = fsc.get_batch(dicebox_config.DATA_DIRECTORY, network_input_index,
                                                       train_batch_size, category_map, noise=noise)
    train_image_data = numpy.array(train_image_data)
    train_image_data = train_image_data.astype('float32')
    train_image_data /= 255
    train_image_labels = numpy.array(train_image_labels)


    test_image_data, test_image_labels = fsc.get_batch(dicebox_config.DATA_DIRECTORY, network_input_index,
                                             test_batch_size, category_map, noise=noise)
    test_image_data = numpy.array(test_image_data)
    test_image_data = test_image_data.astype('float32')
    test_image_data /= 255
    test_image_labels = numpy.array(test_image_labels)

    logging.info("nb_classes: (%i)" % nb_classes)
    logging.info("batch_size: (%i)" % batch_size)
    logging.info("input_shape: (%s)" % input_shape)
    logging.debug("network_input_index: (%s)" % network_input_index)
    logging.info("category_map: (%s)" % category_map)

    #
    #return (nb_classes, batch_size, input_shape, x_test, y_test)
    #return (nb_classes, batch_size, input_shape, image_data, image_labels)
    x_train = train_image_data
    x_test = test_image_data
    y_train = train_image_labels
    y_test = test_image_labels
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_dicebox_filesystem_test():
    nb_classes = 5
    batch_size = 1000
    input_shape = (3000,)

    noise = 0.0
    network_input_index = fsc.get_data_set(dicebox_config.DATA_DIRECTORY)
    category_map = fsc.get_data_set_categories(network_input_index)

    test_image_data, test_image_labels = fsc.get_batch(dicebox_config.DATA_DIRECTORY, network_input_index,
                                             dicebox_config.BATCH_SIZE, category_map, noise=noise)
    test_image_data = numpy.array(test_image_data)
    test_image_data = test_image_data.astype('float32')
    test_image_data /= 255
    test_image_labels = numpy.array(test_image_labels)

    logging.info("nb_classes: (%i)" % nb_classes)
    logging.info("batch_size: (%i)" % batch_size)
    logging.info("input_shape: (%s)" % input_shape)
    # logging.debug("network_input_index: (%s)" % network_input_index)
    logging.info("category_map: (%s)" % category_map)

    #
    #return (nb_classes, batch_size, input_shape, x_test, y_test)
    #return (nb_classes, batch_size, input_shape, image_data, image_labels)
    x_test = test_image_data
    y_test = test_image_labels
    return (nb_classes, batch_size, input_shape, x_test, y_test)

def get_mnist_test():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 1
    input_shape = (784,)

    # Get the data.
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    #x_train = x_train.reshape(60000, 784)
    #x_test = x_test.reshape(10000, 784)
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #logging.info(x_test)
    #logging.info(y_test)

    #x_test = numpy.array([x_test[0]])
    #y_test = numpy.array([y_test[0]])
    #logging.info(x_test)
    #logging.info(y_test)
    #x_test /= 255

    noise = 0.0
    network_input_index = fsc.get_data_set(config.DATA_DIRECTORY)
    category_map = fsc.get_data_set_categories(network_input_index)
    image_data, image_labels = fsc.get_batch(config.DATA_DIRECTORY, network_input_index,
                                             config.BATCH_SIZE, category_map, noise=noise)
    image_data = numpy.array(image_data)
    image_data = image_data.astype('float32')
    image_data /= 255
    image_labels = numpy.array(image_labels)
    # set_image_labels = set(image_labels)
    # logging.info('########################################################################')
    # logging.info(network_input_index)
    # logging.info(category_map)
    # logging.info(image_data)
    # logging.info(image_labels)
    # logging.info('########################################################################')

    # convert class vectors to binary class matrices
    # y_train = to_categorical(y_train, nb_classes)
    # y_test = to_categorical(y_test, nb_classes)

    logging.info("nb_classes: (%i)" % nb_classes)
    logging.info("batch_size: (%i)" % batch_size)
    logging.info("input_shape: (%s)" % input_shape)
    #
    #return (nb_classes, batch_size, input_shape, x_test, y_test)
    return (nb_classes, batch_size, input_shape, image_data, image_labels)

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist_filesystem()
    elif dataset == 'dicebox':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_dicebox_filesystem()

    model = compile_model(network, nb_classes, input_shape)

    ## add some logging
    logging.info('Fitting model:')
    logging.info(network)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

def train_and_score_and_save(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
        x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
        x_test, y_train, y_test = get_mnist_filesystem()
    elif dataset == 'dicebox':
        nb_classes, batch_size, input_shape, x_train, \
        x_test, y_train, y_test = get_dicebox_filesystem()

    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

def load_and_score(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_mnist_filesystem()

    model = compile_model(network, nb_classes, input_shape)

    # load weights
    model.load_weights("weights.best.hdf5")

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.

def load_and_score_single(network, dataset):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, _, \
            x_test, _, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_test, y_test = get_mnist_test()

    model = compile_model(network, nb_classes, input_shape)

    # load weights
    model.load_weights("weights.best.hdf5")

    score = model.evaluate(x_test, y_test, verbose=0)

    model_prediction = model.predict_classes(x_test, batch_size=1, verbose=1)
    logging.info("model_prection")
    logging.info(model_prediction)

    return score[1]  # 1 is accuracy. 0 is loss.

def load_and_predict_single(network, dataset):
    if dataset == 'cifar10':
        nb_classes, batch_size, input_shape, _, \
            x_test, _, y_test = get_cifar10()
    elif dataset == 'mnist':
        nb_classes, batch_size, input_shape, x_test, y_test = get_mnist_test()
    elif dataset == 'dicebox':
        nb_classes, batch_size, input_shape, x_test, y_test = get_dicebox_filesystem_test()

    model = compile_model(network, nb_classes, input_shape)

    # load weights
    model.load_weights("weights.best.hdf5")

    score = model.evaluate(x_test, y_test, verbose=0)

    model_prediction = model.predict_classes(x_test, batch_size=1, verbose=1)
    logging.info("model_prection")
    logging.info(model_prediction)

    return model_prediction