##############################################################################
# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE.txt
##############################################################################

"""Class that represents the network to be evolved."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint
import logging
import numpy
from datetime import datetime
import os
import json

from .config import DiceboxConfig
from .connectors import FileSystemConnector, SensoryServiceConnector
from .utils import random_index, random_index_between, random, random_strict


class DiceboxNetwork:
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """
    fsc = None  # file system connector
    ssc = None  # sensory service connector

    config = None

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

    def __init__(self,
                 nn_param_choices=None,
                 create_fcs=True,
                 disable_data_indexing=False,
                 config_file='./dicebox.config',
                 lonestar_model_file='./dicebox.lonestar.json'):
        if self.config is None:
            self.config = DiceboxConfig(config_file=config_file,
                                        lonestar_model_file=lonestar_model_file)

        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """

        self.accuracy = 0.
        self.network = {}  # (the dicebox object)
        self.model = None # the compiled network.

        if self.fsc is None and create_fcs is True:
            logging.debug('creating a new fsc..')
            logging.info('self.config.DATA_DIRECTORY: (%s)' % self.config.DATA_DIRECTORY)
            self.fsc = FileSystemConnector(data_directory=self.config.DATA_DIRECTORY,
                                           disable_data_indexing=disable_data_indexing,
                                           config_file=config_file,
                                           lonestar_model_file=lonestar_model_file)

        if self.ssc is None:
            logging.debug('creating a new ssc..')
            self.ssc = SensoryServiceConnector(role='client',
                                               config_file=config_file,
                                               lonestar_model_file=lonestar_model_file)

    def create_random(self):
        self.network = {}
        self.network['layers'] = []

        # Set unchange-ables
        self.network['input_shape'] = self.config.INPUT_SHAPE
        self.network['output_size'] = self.config.NB_CLASSES

        # Select an optimizer
        optimizer_index = random_index(len(self.config.TAXONOMY['optimizer']))
        optimizer = self.config.TAXONOMY['optimizer'][optimizer_index - 1]
        self.network['optimizer'] = optimizer

        # Determine the number of layers..
        layer_count = random_index_between(self.config.TAXONOMY['min_layers'],
                                                   self.config.TAXONOMY['max_layers'])
        for layer_index in range(1, layer_count):
            # add new random layer to the network
            self.network['layers'].append(self.build_random_layer())

    def select_random_optimizer(self):
        # Select an optimizer
        optimizer_index = random_index(len(self.config.TAXONOMY['optimizer']))
        return self.config.TAXONOMY['optimizer'][optimizer_index - 1]

    def build_random_layer(self):
        # determine what the layer type will be
        layer_type_index = random_index(len(self.config.TAXONOMY['layer_types']))
        layer_type = self.config.TAXONOMY['layer_types'][layer_type_index - 1]

        random_layer = {}
        random_layer['type'] = layer_type
        if layer_type == 'dropout':
            # get a dropout rate..
            random_layer['rate']: float = random_strict()
        else:
            # determine the size and activation function to use.
            random_layer['size'] = random_index_between(self.config.TAXONOMY['min_neurons'],
                                                                self.config.TAXONOMY['max_neurons'])
            activation_index = random_index(len(self.config.TAXONOMY['activation']))
            random_layer['activation'] = self.config.TAXONOMY['activation'][activation_index - 1]
        return random_layer

    def create_lonestar(self, create_model=False, weights_filename=None):
        logging.debug('-' * 80)
        logging.debug('create_lonestar(create_model, weights_filename)')
        logging.debug(create_model)
        logging.debug(weights_filename)
        logging.debug('-' * 80)

        self.network = self.config.LONESTAR_DICEBOX_MODEL
        logging.debug('-' * 80)
        logging.debug(self.network)
        logging.debug('-' * 80)

        if create_model is True:
            if self.model is None:
                logging.debug('compiling model')
                self.model = DiceboxNetwork.compile_model(self.network)
                if weights_filename is not None:
                    logging.debug("loading weights file: (%s)" % weights_filename)
                    self.load_model(weights_filename)
            # else:
            #     logging.info('model already compiled, skipping.')

    def create_set(self, network):
        self.network = network

    def train(self):
        if self.accuracy == 0.:
            self.accuracy = self.train_and_score(self.network)

    def train_and_save(self, dataset):
        # if self.accuracy == 0.:
        logging.debug('-' * 80)
        logging.debug("train_and_save(dataset)")
        logging.debug("train_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        self.accuracy = self.train_and_score_and_save(dataset)

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

    def train_and_score(self, network):
        if self.config.DICEBOX_COMPLIANT_DATASET is True:
            x_train, x_test, y_train, y_test = self.get_dicebox_filesystem()
        else:
            raise Exception('Unknown dataset type!  Please define, or correct.')

        model = DiceboxNetwork.compile_model(network)

        logging.info('batch size (model.fit): %s' % self.config.BATCH_SIZE)

        logging.info('Fitting network:')
        logging.info(network)
        logging.info('compiled model:')
        logging.info(json.dumps(json.loads(model.to_json())))

        model.fit(x_train, y_train,
                  batch_size=self.config.BATCH_SIZE,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[self.early_stopper])

        score = model.evaluate(x_test, y_test, verbose=0)

        return score[1]  # 1 is accuracy. 0 is loss.

    @staticmethod
    def compile_model(dicebox_model):
        model = Sequential()

        layers = dicebox_model['layers']
        first_layer = False
        for layer in layers:
            # build and add layer
            if layer['type'] == 'dropout':
                # handle dropout
                model.add(Dropout(layer['rate']))
            else:
                neurons = layer['size']
                activation = layer['activation']

                if first_layer is False:
                    first_layer = True
                    model.add(Dense(neurons, activation=activation, input_shape=dicebox_model['input_shape']))
                else:
                    model.add(Dense(neurons, activation=activation))

        # add final output layer.
        model.add(Dense(dicebox_model['output_size'], activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=dicebox_model['optimizer'],
                      metrics=['accuracy'])

        return model

    def get_dicebox_filesystem(self):
        noise = self.config.NOISE
        test_batch_size = self.config.TEST_BATCH_SIZE
        train_batch_size = self.config.TRAIN_BATCH_SIZE

        logging.info('noise: %s' % noise)
        logging.info('train_batch_size: %s' % train_batch_size)
        logging.info('test_batch_size: %s' % test_batch_size)

        train_image_data, train_image_labels = self.fsc.get_batch(train_batch_size, noise=noise)
        # train_image_data, train_image_labels = Network.ssc.get_batch(train_batch_size, noise=noise)
        train_image_data = numpy.array(train_image_data)
        train_image_data = train_image_data.astype('float32')
        train_image_data /= 255
        train_image_labels = numpy.array(train_image_labels)

        test_image_data, test_image_labels = self.fsc.get_batch(test_batch_size, noise=noise)
        # test_image_data, test_image_labels = Network.ssc.get_batch(test_batch_size, noise=noise)
        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255
        test_image_labels = numpy.array(test_image_labels)

        x_train = train_image_data
        x_test = test_image_data
        y_train = train_image_labels
        y_test = test_image_labels

        return x_train, x_test, y_train, y_test

    def get_dicebox_sensory_data(self):
        logging.debug('-' * 80)
        logging.debug('get_dicebox_sensory_data(self)')
        logging.debug('-' * 80)

        noise = self.config.NOISE
        train_batch_size = self.config.TRAIN_BATCH_SIZE
        test_batch_size = self.config.TEST_BATCH_SIZE

        try:
            # train_image_data, train_image_labels = Network.fsc.get_batch(train_batch_size, noise=noise)
            train_image_data, train_image_labels = self.ssc.get_batch(train_batch_size, noise=noise)

            logging.debug('-' * 80)
            logging.debug('train_image_data to numpy.array')
            # logging.debug(train_image_data)

            train_image_data = numpy.array(train_image_data)
            # logging.debug(train_image_data)

            logging.debug('train_image_data astype float32')
            train_image_data = train_image_data.astype('float32')
            # logging.debug(train_image_data)

            logging.debug('train_image_data /255')
            train_image_data /= 255
            # logging.debug(train_image_data)

            logging.debug('train_image_labels to numpy.array')
            train_image_labels = numpy.array(train_image_labels)
            # logging.debug(train_image_labels)
            logging.debug('-' * 80)
        except ValueError:
            logging.debug('Caught ValueError when processing training data.')
            logging.debug('failing out..')
            raise ValueError

        try:
            # test_image_data, test_image_labels = Network.fsc.get_batch(test_batch_size, noise=noise)
            test_image_data, test_image_labels = self.ssc.get_batch(test_batch_size, noise=noise)

            logging.debug('-' * 80)
            logging.debug('test_image_data to numpy.array')
            # logging.debug(test_image_data)

            test_image_data = numpy.array(test_image_data)
            # logging.debug(test_image_data)

            logging.debug('test_image_data astype float32')
            test_image_data = test_image_data.astype('float32')
            # logging.debug(test_image_data)

            logging.debug('test_image_data /255')
            test_image_data /= 255
            # logging.debug(test_image_data)

            logging.debug('test_image_labels to numpy.array')
            test_image_labels = numpy.array(test_image_labels)
            # logging.debug(test_image_labels)
        except ValueError:
            logging.debug('Caught ValueError when processing test data.')
            logging.debug('failing out..')
            raise ValueError

        x_train = train_image_data
        x_test = test_image_data
        y_train = train_image_labels
        y_test = test_image_labels
        return x_train, x_test, y_train, y_test

    def train_and_score_and_save(self, dataset):
        logging.debug('-' * 80)
        logging.debug("train_and_score_and_save(dataset)")
        logging.debug("train_and_score_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        if self.config.DICEBOX_COMPLIANT_DATASET is True:
            logging.debug('-' * 80)
            logging.debug('loading sensory data..')
            logging.debug('-' * 80)
            x_train, x_test, y_train, y_test = self.get_dicebox_sensory_data()
            # nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_dicebox_sensory_data()
            # nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = self.get_dicebox_filesystem()
            logging.debug('-' * 80)
            logging.debug('Done!')
            logging.debug('-' * 80)
        else:
            # no support yet!
            logging.error('UNSUPPORTED dataset supplied to train_and_score_and_save')
            raise Exception('UNSUPPORTED dataset supplied to train_and_score_and_save')

        logging.debug('-' * 80)
        logging.debug('Compiling model if need be.')
        logging.debug('-' * 80)
        if self.model is None:
            self.model = DiceboxNetwork.compile_model(self.network)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        logging.debug('-' * 80)
        logging.debug('Fitting model.')
        logging.debug('-' * 80)
        self.model.fit(x_train, y_train,
                       batch_size=self.config.BATCH_SIZE,
                       epochs=10000,  # using early stopping, so no real limit
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=self.callbacks_list)
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
        if self.model is None:
            logging.error('no model! :(  compile the model first.')
            raise Exception('no model! :(  compile the model first.')
        logging.debug('loading weights file..')
        try:
            self.model.load_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to load weights file.')
            logging.error(e)
            raise e

    def load_model(self, filename):
        if self.model is None:
            logging.error('no model! :(  compile the model first.')
            raise Exception('no model! :(  compile the model first.')
        logging.debug('loading weights file..')
        try:
            self.model.load_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to load weights file.')
            logging.error(e)
            raise e
        raise Exception('Not yet implemented!')

    def classify(self, network_input):
        if self.config.DICEBOX_COMPLIANT_DATASET is True:
            x_test = self.get_dicebox_raw(network_input)
        else:
            logging.error("UNKNOWN DATASET (%s) passed to classify" % self.config.NETWORK_NAME)
            raise Exception("UNKNOWN DATASET (%s) passed to classify" % self.config.NETWORK_NAME)

        if self.model is None:
            logging.error('Unable to classify without a model. :(')
            raise Exception('Unable to classify without a model. :(')

        model_prediction = self.model.predict_classes(x_test, batch_size=1, verbose=0)
        logging.info(model_prediction)

        return model_prediction

    def get_dicebox_raw(self, raw_image_data):
        # ugh dump to file for the time being
        filename = "%s/%s" % (self.config.TMP_DIR, datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png'))
        with open(filename, 'wb') as f:
            f.write(raw_image_data)

        try:
            test_image_data = self.fsc.process_image(filename)
        except:
            logging.error('Exception caught processing image data.')
            raise Exception('Exception caught processing image data.')
        finally:
            os.remove(filename)

        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255

        x_test = [test_image_data]
        x_test = numpy.array(x_test)

        return x_test
