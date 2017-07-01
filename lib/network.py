"""Class that represents the network to be evolved."""
import random
import logging
#from train import train_and_score

#from train import train_and_score_and_save
#from train import load_and_score
#from train import load_and_score_single
#from train import load_and_predict_single

#import filesystem_connecter as fsc

from keras.callbacks import EarlyStopping
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import logging
import numpy
import dicebox_config as config
import filesystem_connecter
from datetime import datetime
import os

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=25)

# Checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [early_stopper, checkpoint]


class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """
    fsc = None

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.model = None

        if Network.fsc is None:
            #logging.debug('creating a new fsc..')
            Network.fsc = filesystem_connecter.FileSystemConnector(config.DATA_DIRECTORY)

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_lonestar(self, create_model=False, weights_filename=None):

        # mnist
        # Lonestar /mnist approx 98.6% acc
        # {'nb_layers': 2, 'activation': 'relu', 'optimizer': 'adagrad', 'nb_neurons': 1597}
        # self.network['nb_layers'] = 2
        # self.network['activation'] = 'relu'
        # self.network['optimizer'] = 'adagrad'
        # self.network['nb_neurons'] = 1597

        # dicebox 60x50
        # Network accuracy: 80.50%
        # {'nb_layers': 1, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 55}
        # self.network['nb_layers'] = 1
        # self.network['activation'] = 'sigmoid'
        # self.network['optimizer'] = 'adamax'
        # self.network['nb_neurons'] = 55

        # dicebox_60x50
        # {'nb_layers': 2, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 610}
        # Network accuracy: 92.20 %
        self.network['nb_layers'] = 2
        self.network['activation'] = 'sigmoid'
        self.network['optimizer'] = 'adamax'
        self.network['nb_neurons'] = 610

        if create_model is True:
            if self.model is None:
                logging.info('compiling model')
                self.model = self.compile_model(self.network, config.NB_CLASSES, config.INPUT_SHAPE)
                if weights_filename is not None:
                    logging.info("loading weights file: (%s)" % weights_filename)
                    self.load_model(weights_filename)
            # else:
            #     logging.info('model already compiled, skipping.')

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = self.train_and_score(self.network, dataset)

    def train_and_save(self, dataset):
    #    if self.accuracy == 0.:
        self.accuracy = self.train_and_score_and_save(dataset)

    # def load_n_score(self, dataset):
    #     self.accuracy = load_and_score(self.network, dataset)

    # def load_n_score_single(self, dataset):
    #     self.accuracy = load_and_score_single(self.network, dataset)

    # def load_n_predict_single(self, dataset, network_input):
    #     return load_and_predict_single(self.network, dataset, network_input)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))


#####################################################################################
#
#
#####################################################################################


    def train_and_score(self, network, dataset):

        """
        Train the model, return test loss.
        Args:
            network (dict): the parameters of the network
            dataset (str): Dataset to use for training/evaluating
        """

        # if dataset == 'cifar10':
        #     nb_classes, batch_size, input_shape, x_train, \
        #         x_test, y_train, y_test = get_cifar10()
        # elif dataset == 'mnist':
        #     nb_classes, batch_size, input_shape, x_train, \
        #         x_test, y_train, y_test = get_mnist_filesystem()
        # el

        if dataset == 'dicebox':
            nb_classes, batch_size, input_shape, x_train, \
                x_test, y_train, y_test = self.get_dicebox_filesystem()

        model = self.compile_model(network, nb_classes, input_shape)

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


    def compile_model(self, network, nb_classes, input_shape):
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


    def get_dicebox_filesystem(self):

        nb_classes = config.NB_CLASSES
        batch_size = config.BATCH_SIZE
        input_shape = config.INPUT_SHAPE
        noise = config.NOISE
        train_batch_size = config.TRAIN_BATCH_SIZE
        test_batch_size = config.TEST_BATCH_SIZE

        train_image_data, train_image_labels = Network.fsc.get_batch(train_batch_size, noise=noise)
        train_image_data = numpy.array(train_image_data)
        train_image_data = train_image_data.astype('float32')
        train_image_data /= 255
        train_image_labels = numpy.array(train_image_labels)

        test_image_data, test_image_labels = Network.fsc.get_batch(test_batch_size, noise=noise)
        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255
        test_image_labels = numpy.array(test_image_labels)

        logging.info("nb_classes: (%i)" % nb_classes)
        logging.info("batch_size: (%i)" % batch_size)
        logging.info("input_shape: (%s)" % input_shape)

        #
        #return (nb_classes, batch_size, input_shape, x_test, y_test)
        #return (nb_classes, batch_size, input_shape, image_data, image_labels)
        x_train = train_image_data
        x_test = test_image_data
        y_train = train_image_labels
        y_test = test_image_labels
        return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


    def train_and_score_and_save(self, dataset):
        """Train the model, return test loss.

        Args:
            network (dict): the parameters of the network
            dataset (str): Dataset to use for training/evaluating

        """
        #if dataset == 'cifar10':
        #    nb_classes, batch_size, input_shape, x_train, \
        #    x_test, y_train, y_test = get_cifar10()
        #elif dataset == 'mnist':
        #    nb_classes, batch_size, input_shape, x_train, \
        #    x_test, y_train, y_test = get_mnist_filesystem()
        #el

        if dataset == 'dicebox':
            nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = self.get_dicebox_filesystem()
        else:
            # no support yet!
            logging.error('UNSUPPORTED dataset supplied to train_and_score_and_save')
            raise

        if self.model is None:
            self.model = self.compile_model(self.network, nb_classes, input_shape)

        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=10000,  # using early stopping, so no real limit
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=callbacks_list)

        score = self.model.evaluate(x_test, y_test, verbose=1)

        return score[1]  # 1 is accuracy. 0 is loss.


    def save_model(self, filename):
        logging.debug('saving model weights to file..')
        self.model.save(filename)


    def load_model(self, filename):
        if self.model is None:
            logging.error('no model! :(  compile the model first.')
            raise
        logging.debug('loading weights file..')
        self.model.load_weights(filename)


    def predict(self, dataset, network_input):
        # if dataset == 'cifar10':
        #     nb_classes, batch_size, input_shape, _, \
        #     x_test, _, y_test = get_cifar10()
        # elif dataset == 'mnist':
        #     nb_classes, batch_size, input_shape, x_test, y_test = get_mnist_test()
        # elif dataset == 'dicebox':
        #     nb_classes, batch_size, input_shape, x_test, y_test = get_dicebox_filesystem_test()
        # el

        if dataset == 'dicebox_raw':
            x_test = self.get_dicebox_raw(network_input)
        else:
            logging.error("UNKNOWN DATASET (%s) passed to predict_single" % dataset)
            raise

        #model = compile_model(network, nb_classes, input_shape)
        if self.model is None:
            #self.model = self.compile_model(self.network, nb_classes, input_shape)
            logging.error('Unable to predict without a model. :(')
            raise

            # # load weights
            # if filename is not None:
            #     self.load_model(filename)

        # load weights
        #self.model.load_weights("weights.best.hdf5")

        #logging.info(x_test)

        # score = model.evaluate(x_test, y_test, verbose=0)

        model_prediction = self.model.predict_classes(x_test, batch_size=1, verbose=0)
        # logging.info("model_prection")
        logging.info(model_prediction)

        return model_prediction


    def get_dicebox_raw(self, raw_image_data):
        # ugh dump to file for the time being
        filename = "./tmp/%s" % datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png')
        with open(filename, 'wb') as f:
            f.write(raw_image_data)

        test_image_data = self.fsc.process_image(filename)

        os.remove(filename)

        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255

        #logging.info("nb_classes: (%i)" % nb_classes)
        #logging.info("batch_size: (%i)" % batch_size)
        #logging.info("input_shape: (%s)" % input_shape)

        x_test = [test_image_data]
        x_test = numpy.array(x_test)

        #logging.info("x_test: (%s)" % x_test)
        return x_test
