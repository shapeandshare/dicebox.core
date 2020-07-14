# from tensorflow.keras.callbacks import ModelCheckpoint
import logging
import os
from datetime import datetime
from typing import Union, Any, List

import numpy
from numpy import ndarray
from tensorflow.keras.callbacks import EarlyStopping

from .layer import DropoutLayer, DenseLayer
from .optimizers import Optimizers
from ..config.dicebox_config import DiceboxConfig
from ..connectors.filesystem_connector import FileSystemConnector
from ..connectors.sensory_service_connector import SensoryServiceConnector
from .network import Network


class DiceboxNetwork(Network):
    __accuracy: float

    __fsc: FileSystemConnector  # file system connector
    __ssc: SensoryServiceConnector  # sensory service connector

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
                 input_shape: int,
                 output_size: int,
                 optimizer: Optimizers,
                 layers: List[Union[DropoutLayer, DenseLayer]] = None,
                 create_fsc: bool = True,
                 disable_data_indexing: bool = False):

        super().__init__(config=config, input_shape=input_shape, output_size=output_size, optimizer=optimizer, layers=layers)

        self.__accuracy: float = 0.0

        if create_fsc is True:
            logging.debug('creating a new fsc..')
            logging.info('self.config.DATA_DIRECTORY: (%s)' % self.config.DATA_DIRECTORY)
            self.__fsc = FileSystemConnector(config=config,
                                             data_directory=self.config.DATA_DIRECTORY,
                                             disable_data_indexing=disable_data_indexing)
        else:
            logging.debug('creating a new ssc..')
            self.__ssc = SensoryServiceConnector(role='client', config=self.config)

    ## Training

    def train(self, update_accuracy=False) -> float:
        if self.config.DICEBOX_COMPLIANT_DATASET is True:
            x_train, x_test, y_train, y_test = self.get_dicebox_filesystem()
        else:
            raise Exception('Unknown dataset type!  Please define, or correct.')

        self.compile()

        self.model.fit(x_train, y_train,
                       batch_size=self.config.BATCH_SIZE,
                       epochs=self.config.EPOCHS, # using early stopping, so this limit acts like a max
                       verbose=1,
                       validation_data=(x_test, y_test),
                       callbacks=[self.__early_stopper])

        score = self.model.evaluate(x_test, y_test, verbose=0)

        if update_accuracy is True:
            self.__accuracy = score

        return score[1]  # 1 is accuracy. 0 is loss.

    def get_accuracy(self) -> float:
        return self.__accuracy

    ## Prediction

    def classify(self, network_input: Any) -> ndarray:
        if self.config.DICEBOX_COMPLIANT_DATASET is True:
            x_test: ndarray = self.__get_dicebox_raw(network_input)
        else:
            logging.error("UNKNOWN DATASET (%s) passed to classify" % self.config.NETWORK_NAME)
            raise Exception("UNKNOWN DATASET (%s) passed to classify" % self.config.NETWORK_NAME)

        if self.model is None:
            logging.error('No model! Compile the network first.')
            raise Exception('No model! Compile the network first.')

        model_prediction: ndarray = self.model.predict_classes(x_test, batch_size=1, verbose=0)
        logging.info(model_prediction)

        return model_prediction

    ## Weights Storage Functions

    def save_model_weights(self, filename: str) -> None:
        if self.model is None:
            logging.error('No model! Compile the network first.')
            raise Exception('No model! Compile the network first.')

        logging.debug('loading weights file..')
        try:
            self.model.save_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to save weights file.')
            logging.error(e)
            raise e

    def load_model_weights(self, filename: str) -> None:
        if self.model is None:
            logging.error('No model! Compile the network first.')
            raise Exception('No model! Compile the network first.')

        logging.debug('loading weights file..')
        try:
            self.model.load_weights(str(filename))  # https://github.com/keras-team/keras/issues/11269
        except Exception as e:
            logging.error('Unable to load weights file.')
            logging.error(e)
            raise e

    ## Data Centric Functions

    def __get_dicebox_raw(self, raw_image_data: Any) -> ndarray:
        # TODO: variable reuse needs to be cleaned up..

        # ugh dump to file for the time being
        filename = "%s/%s" % (self.config.TMP_DIR, datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png'))
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
        noise = self.config.NOISE
        test_batch_size = self.config.TEST_BATCH_SIZE
        train_batch_size = self.config.TRAIN_BATCH_SIZE

        logging.debug('noise: %s' % noise)
        logging.debug('train_batch_size: %s' % train_batch_size)
        logging.debug('test_batch_size: %s' % test_batch_size)

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

        noise = self.config.NOISE
        train_batch_size = self.config.TRAIN_BATCH_SIZE
        test_batch_size = self.config.TEST_BATCH_SIZE

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
