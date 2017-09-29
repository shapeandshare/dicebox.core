##############################################################################
# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE
##############################################################################

"""Class that represents the network to be evolved."""
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
import logging
import numpy
import docker_config as config
import filesystem_connecter
import sensory_interface
from datetime import datetime
import os

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=25)

##############################################################################
# Feature disabled until a flipper can be added and the filenames created safely.
# Since this now runs in a container some additional considerations must be made.
##############################################################################
# Checkpoint
# filepath = "%s/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" % config.WEIGHTS_DIR
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [early_stopper, checkpoint]

callbacks_list = [early_stopper]


class Network:
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """
    fsc = None  # file system connector
    ssc = None  # sensory service connector

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
            # logging.debug('creating a new fsc..')
            Network.fsc = filesystem_connecter.FileSystemConnector(config.DATA_DIRECTORY)

        if Network.ssc is None:
            logging.debug('creating a new ssc..')
            Network.ssc = sensory_interface.SensoryInterface('client')

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_lonestar(self, create_model=False, weights_filename=None):
        logging.debug('-' * 80)
        logging.debug('create_lonestar(create_model, weights_filename)')
        logging.debug(create_model)
        logging.debug(weights_filename)
        logging.debug('-' * 80)

        # Load from external definition
        self.network['nb_layers'] = config.NN_LONESTAR_PARAMS['nb_layers']
        self.network['activation'] = config.NN_LONESTAR_PARAMS['activation']
        self.network['optimizer'] = config.NN_LONESTAR_PARAMS['optimizer']
        self.network['nb_neurons'] = config.NN_LONESTAR_PARAMS['nb_neurons']
        logging.debug('-' * 80)
        logging.debug("self.network['nb_layers']: %s" % self.network['nb_layers'])
        logging.debug("self.network['activation']: %s" % self.network['activation'])
        logging.debug("self.network['optimizer']: %s" % self.network['optimizer'])
        logging.debug("self.network['nb_neurons']:%s" % self.network['nb_neurons'])
        logging.debug('-' * 80)

        if create_model is True:
            if self.model is None:
                logging.debug('compiling model')
                self.model = self.compile_model(self.network, config.NB_CLASSES, config.INPUT_SHAPE)
                if weights_filename is not None:
                    logging.debug("loading weights file: (%s)" % weights_filename)
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
        # if self.accuracy == 0.:
        logging.debug('-' * 80)
        logging.debug("train_and_save(dataset)")
        logging.debug("train_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        self.accuracy = self.train_and_score_and_save(dataset)

    def print_network(self):
        """Print out a network."""
        logging.debug(self.network)
        logging.debug("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def train_and_score(self, network, dataset):
        if dataset == 'dicebox':
            nb_classes, batch_size, input_shape, x_train, \
                x_test, y_train, y_test = self.get_dicebox_filesystem()
        else:
            raise

        model = self.compile_model(network, nb_classes, input_shape)

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
        # train_image_data, train_image_labels = Network.ssc.get_batch(train_batch_size, noise=noise)
        train_image_data = numpy.array(train_image_data)
        train_image_data = train_image_data.astype('float32')
        train_image_data /= 255
        train_image_labels = numpy.array(train_image_labels)

        test_image_data, test_image_labels = Network.fsc.get_batch(test_batch_size, noise=noise)
        # test_image_data, test_image_labels = Network.ssc.get_batch(test_batch_size, noise=noise)
        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255
        test_image_labels = numpy.array(test_image_labels)

        logging.debug("nb_classes: (%i)" % nb_classes)
        logging.debug("batch_size: (%i)" % batch_size)
        logging.debug("input_shape: (%s)" % input_shape)

        x_train = train_image_data
        x_test = test_image_data
        y_train = train_image_labels
        y_test = test_image_labels
        return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test

    def get_dicebox_sensory_data(self):
        logging.debug('-' * 80)
        logging.debug('get_dicebox_sensory_data(self)')
        logging.debug('-' * 80)

        nb_classes = config.NB_CLASSES
        batch_size = config.BATCH_SIZE
        input_shape = config.INPUT_SHAPE
        noise = config.NOISE
        train_batch_size = config.TRAIN_BATCH_SIZE
        test_batch_size = config.TEST_BATCH_SIZE

        # train_image_data, train_image_labels = Network.fsc.get_batch(train_batch_size, noise=noise)
        train_image_data, train_image_labels = Network.ssc.get_batch(train_batch_size, noise=noise)
        try:
            logging.debug('-' * 80)
            logging.debug('train_image_data to numpy.array')
            #logging.debug(train_image_data)

            train_image_data = numpy.array(train_image_data)
            #logging.debug(train_image_data)

            logging.debug('train_image_data astype float32')
            train_image_data = train_image_data.astype('float32')
            #logging.debug(train_image_data)

            logging.debug('train_image_data /255')
            train_image_data /= 255
            #logging.debug(train_image_data)

            logging.debug('train_image_labels to numpy.array')
            train_image_labels = numpy.array(train_image_labels)
            #logging.debug(train_image_labels)
            logging.debug('-' * 80)
        except ValueError:
            logging.debug('Caught ValueError when processing training data.')
            logging.debug('failing out..')
            raise

        # test_image_data, test_image_labels = Network.fsc.get_batch(test_batch_size, noise=noise)
        test_image_data, test_image_labels = Network.ssc.get_batch(test_batch_size, noise=noise)
        try:
            logging.debug('-' * 80)
            logging.debug('test_image_data to numpy.array')
            #logging.debug(test_image_data)

            test_image_data = numpy.array(test_image_data)
            #logging.debug(test_image_data)

            logging.debug('test_image_data astype float32')
            test_image_data = test_image_data.astype('float32')
            #logging.debug(test_image_data)

            logging.debug('test_image_data /255')
            test_image_data /= 255
            #logging.debug(test_image_data)

            logging.debug('test_image_labels to numpy.array')
            test_image_labels = numpy.array(test_image_labels)
            #logging.debug(test_image_labels)
        except ValueError:
            logging.debug('Caught ValueError when processing test data.')
            logging.debug('failing out..')
            raise

        logging.debug("nb_classes: (%i)" % nb_classes)
        logging.debug("batch_size: (%i)" % batch_size)
        logging.debug("input_shape: (%s)" % input_shape)

        x_train = train_image_data
        x_test = test_image_data
        y_train = train_image_labels
        y_test = test_image_labels
        return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test

    def train_and_score_and_save(self, dataset):
        logging.debug('-' * 80)
        logging.debug("train_and_score_and_save(dataset)")
        logging.debug("train_and_score_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        if dataset == 'dicebox':
            # nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_dicebox_filesystem()
            logging.debug('-' * 80)
            logging.debug('loading sensory data..')
            logging.debug('-' * 80)
            nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_dicebox_sensory_data()
            logging.debug('-' * 80)
            logging.debug('Done!')
            logging.debug('-' * 80)
        else:
            # no support yet!
            logging.error('UNSUPPORTED dataset supplied to train_and_score_and_save')
            raise

        logging.debug('-' * 80)
        logging.debug('Compiling mode if need be.')
        logging.debug('-' * 80)
        if self.model is None:
            self.model = self.compile_model(self.network, nb_classes, input_shape)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        logging.debug('-' * 80)
        logging.debug('Fitting model.')
        logging.debug('-' * 80)
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=10000,  # using early stopping, so no real limit
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=callbacks_list)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        logging.debug('-' * 80)
        logging.debug('Scoring model.')
        logging.debug('-' * 80)
        score = self.model.evaluate(x_test, y_test, verbose=1)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        return score[1]  # 1 is accuracy. 0 is loss.

    def save_model(self, filename):
        logging.debug('saving model weights to file..')
        self.model.save(filename)

    def load_model(self, filename):
        if self.model is None:
            logging.error('no model! :(  compile the model first.')
            raise
        logging.debug('loading weights file..')
        try:
            self.model.load_weights(filename)
        except Exception as e:
            logging.error('Unable to load weights file.')
            logging.error(e)
            raise

    def classify(self, dataset, network_input):
        if dataset == 'dicebox_raw':
            x_test = self.get_dicebox_raw(network_input)
        else:
            logging.error("UNKNOWN DATASET (%s) passed to classify" % dataset)
            raise

        if self.model is None:
            logging.error('Unable to classify without a model. :(')
            raise

        model_prediction = self.model.predict_classes(x_test, batch_size=1, verbose=0)
        logging.info(model_prediction)

        return model_prediction

    def get_dicebox_raw(self, raw_image_data):
        # ugh dump to file for the time being
        filename = "%s/%s" % (config.TMP_DIR, datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png'))
        with open(filename, 'wb') as f:
            f.write(raw_image_data)

        try:
            test_image_data = self.fsc.process_image(filename)
        except:
            os.remove(filename)
            logging.error('Exception caught processing image data.')
            raise

        os.remove(filename)

        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255

        x_test = [test_image_data]
        x_test = numpy.array(x_test)

        return x_test
