##############################################################################
# Derived from https://github.com/harvitronix/neural-network-genetic-algorithm
# Derived source copyright: Matt Harvey, 2017, Derived source license: The MIT License
# See docs/Matt Harvey.LICENSE.txt
##############################################################################

"""Class that represents the network to be evolved."""
from typing import Union, Any

from numpy import ndarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import ModelCheckpoint
import logging
import numpy
from datetime import datetime
import os
import json

from .config import DiceboxConfig
from .connectors import FileSystemConnector, SensoryServiceConnector
from .layer_factory import LayerFactory
from .models.layer import DropoutLayer, DenseLayer, DropoutLayerConfigure, DenseLayerConfigure, LayerType
from .models.network import Network, Optimizers
from .network_factory import NetworkFactory


class DiceboxNetwork:
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """
    __fsc: FileSystemConnector  # file system connector
    __ssc: SensoryServiceConnector  # sensory service connector

    __config: DiceboxConfig

    __network_factory: NetworkFactory

    # Helper: Early stopping.
    __early_stopper = EarlyStopping(patience=25)

    ##############################################################################
    # Feature disabled until a flipper can be added and the filenames created safely.
    # Since this now runs in a container some additional considerations must be made.
    ##############################################################################
    # Checkpoint
    # filepath = "%s/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" % __config.WEIGHTS_DIR
    # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # __callbacks_list = [__early_stopper, checkpoint]

    __callbacks_list = [__early_stopper]

    def __init__(self,
                 config: DiceboxConfig,
                 create_fsc: bool = True,
                 disable_data_indexing: bool = False):

        self.__config = config

        self.__accuracy: float = 0.0

        self.__network_factory = NetworkFactory(config=self.__config)
        self.__network: Union[Network, None] = None   # the network object
        # self.__model: Union[Sequential, None] = None  # the compiled network (model).

        if create_fsc is True:
            logging.debug('creating a new fsc..')
            logging.info('self.config.DATA_DIRECTORY: (%s)' % self.__config.DATA_DIRECTORY)
            self.__fsc = FileSystemConnector(data_directory=self.__config.DATA_DIRECTORY,
                                             config=self.__config,
                                             disable_data_indexing=disable_data_indexing)
        else:
            logging.debug('creating a new ssc..')
            self.__ssc = SensoryServiceConnector(role='client', config_file=config_file)

    # def create_set(self, __network: Network) -> None:
    #     self.__network: Network = __network

    ## Logging

    def print_network(self) -> None:
        """Print out a __network."""
        logging.info(self.__network)
        logging.info("Network accuracy: %.2f%%" % (self.__accuracy * 100))


    ## Training

    def train(self) -> None:
        if self.__accuracy == 0.:
            self.__accuracy = self.train_and_score(self.__network)

    def train_and_save(self, dataset: Any) -> None:
        # if self.__accuracy == 0.:
        logging.debug('-' * 80)
        logging.debug("train_and_save(dataset)")
        logging.debug("train_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        self.__accuracy = self.train_and_score_and_save(dataset)

    # does not update the instance model or network
    def train_and_score(self, network: Network) -> float:
        if self.__config.DICEBOX_COMPLIANT_DATASET is True:
            x_train, x_test, y_train, y_test = self.get_dicebox_filesystem()
        else:
            raise Exception('Unknown dataset type!  Please define, or correct.')

        model = self.__network.compile()

        logging.info('batch size (model.fit): %s' % self.__config.BATCH_SIZE)

        logging.info('Fitting network:')
        logging.info(network)
        logging.info('compiled model:')
        logging.info(json.dumps(json.loads(model.to_json())))

        model.fit(x_train, y_train,
                  batch_size=self.__config.BATCH_SIZE,
                  epochs=10000,  # using early stopping, so no real limit
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[self.__early_stopper])

        score = model.evaluate(x_test, y_test, verbose=0)

        return score[1]  # 1 is accuracy. 0 is loss.

    # updates the instance's model and network
    def train_and_score_and_save(self, dataset: Any) -> float:
        logging.debug('-' * 80)
        logging.debug("train_and_score_and_save(dataset)")
        logging.debug("train_and_score_and_save(dataset=%s)" % dataset)
        logging.debug('-' * 80)
        if self.__config.DICEBOX_COMPLIANT_DATASET is True:
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
        if self.__network.model is None:
            self.__network.compile()
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        logging.debug('-' * 80)
        logging.debug('Fitting model.')
        logging.debug('-' * 80)
        self.__network.model.fit(x_train, y_train,
                         batch_size=self.__config.BATCH_SIZE,
                         epochs=10000,  # using early stopping, so no real limit
                         verbose=1,
                         validation_data=(x_test, y_test),
                         callbacks=self.__callbacks_list)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        logging.debug('-' * 80)
        logging.debug('Scoring model.')
        logging.debug('-' * 80)
        score = self.__network.model.evaluate(x_test, y_test, verbose=1)
        logging.debug('-' * 80)
        logging.debug('Done!')
        logging.debug('-' * 80)

        return score[1]  # 1 is accuracy. 0 is loss.

    def accuracy(self) -> float:
        return self.__accuracy

    ## Prediction

    def classify(self, network_input: Any) -> ndarray:
        if self.__config.DICEBOX_COMPLIANT_DATASET is True:
            x_test: ndarray = self.__get_dicebox_raw(network_input)
        else:
            logging.error("UNKNOWN DATASET (%s) passed to classify" % self.__config.NETWORK_NAME)
            raise Exception("UNKNOWN DATASET (%s) passed to classify" % self.__config.NETWORK_NAME)

        if self.__network.model is None:
            logging.error('Unable to classify without a model.')
            raise Exception('Unable to classify without a model.')

        model_prediction: ndarray = self.__network.model.predict_classes(x_test, batch_size=1, verbose=0)
        logging.info(model_prediction)

        return model_prediction


    ## Weights Storage Functions

    def save_model_weights(self, filename: str) -> None:
        if self.__network.model is None:
            logging.error('No model! Compile the network first.')
            raise Exception('No model! Compile the network first.')

        logging.debug('loading weights file..')
        try:
            self.__network.model.save_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to save weights file.')
            logging.error(e)
            raise e

    def load_model_weights(self, filename: str) -> None:
        if self.__network.model is None:
            logging.error('No model! Compile the network first.')
            raise Exception('No model! Compile the network first.')

        logging.debug('loading weights file..')
        try:
            self.__network.model.load_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to load weights file.')
            logging.error(e)
            raise e


    ## Data Centric Functions

    def __get_dicebox_raw(self, raw_image_data: Any) -> ndarray:
        # TODO: variable reuse needs to be cleaned up..

        # ugh dump to file for the time being
        filename = "%s/%s" % (self.__config.TMP_DIR, datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png'))
        with open(filename, 'wb') as f:
            f.write(raw_image_data)

        try:
            test_image_data = self.__fsc.process_image(filename)
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

    def get_dicebox_filesystem(self) -> [ndarray, ndarray, ndarray, ndarray]:
        noise = self.__config.NOISE
        test_batch_size = self.__config.TEST_BATCH_SIZE
        train_batch_size = self.__config.TRAIN_BATCH_SIZE

        logging.info('noise: %s' % noise)
        logging.info('train_batch_size: %s' % train_batch_size)
        logging.info('test_batch_size: %s' % test_batch_size)

        train_image_data, train_image_labels = self.__fsc.get_batch(train_batch_size, noise=noise)
        # train_image_data, train_image_labels = Network.__ssc.get_batch(train_batch_size, noise=noise)
        train_image_data = numpy.array(train_image_data)
        train_image_data = train_image_data.astype('float32')
        train_image_data /= 255
        train_image_labels = numpy.array(train_image_labels)

        test_image_data, test_image_labels = self.__fsc.get_batch(test_batch_size, noise=noise)
        # test_image_data, test_image_labels = Network.__ssc.get_batch(test_batch_size, noise=noise)
        test_image_data = numpy.array(test_image_data)
        test_image_data = test_image_data.astype('float32')
        test_image_data /= 255
        test_image_labels = numpy.array(test_image_labels)

        x_train: ndarray = train_image_data
        x_test: ndarray = test_image_data
        y_train: ndarray = train_image_labels
        y_test: ndarray = test_image_labels

        return x_train, x_test, y_train, y_test

    def get_dicebox_sensory_data(self) -> [ndarray, ndarray, ndarray, ndarray]:
        logging.debug('-' * 80)
        logging.debug('get_dicebox_sensory_data(self)')
        logging.debug('-' * 80)

        noise = self.__config.NOISE
        train_batch_size = self.__config.TRAIN_BATCH_SIZE
        test_batch_size = self.__config.TEST_BATCH_SIZE

        try:
            # train_image_data, train_image_labels = Network.__fsc.get_batch(train_batch_size, noise=noise)
            train_image_data, train_image_labels = self.__ssc.get_batch(train_batch_size, noise=noise)

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
            # test_image_data, test_image_labels = Network.__fsc.get_batch(test_batch_size, noise=noise)
            test_image_data, test_image_labels = self.__ssc.get_batch(test_batch_size, noise=noise)

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

        x_train: ndarray = train_image_data
        x_test: ndarray = test_image_data
        y_train: ndarray = train_image_labels
        y_test: ndarray = test_image_labels
        return x_train, x_test, y_train, y_test


    ## Network Functions

    def load_network(self, network_definition: Any) -> None:
        self.__network = self.__network_factory.create_network(network_definition=network_definition)

    def generate_random_network(self) -> None:
        self.__network = self.__network_factory.create_random_network()

    ## For Evolutionary Optimizer

    def get_optimizer(self) -> Optimizers:
        return self.__network.optimizer

    def get_layer_count(self) -> int:
        return len(self.__network.layers)

    def get_layer_definition(self, layer_index) -> Any:
        layer: Union[DropoutLayer, DenseLayer] = self.__network.get_layer_definition(layer_index)
        layer_factory: LayerFactory = LayerFactory(config=self.__config)
        layer_config: Union[DenseLayerConfigure, DropoutLayerConfigure] = layer_factory.decompile_layer(layer)

        definition = {}

        if layer_config.layer_type == LayerType.DROPOUT:
            definition['type'] = LayerType.DROPOUT.value
            definition['rate'] = layer_config.rate
        elif layer_config.layer_type == LayerType.DENSE:
            definition['type'] = LayerType.DENSE.value
            definition['size'] = layer_config.size
            definition['activation'] = layer_config.activation.value
        else:
            raise

        return definition
