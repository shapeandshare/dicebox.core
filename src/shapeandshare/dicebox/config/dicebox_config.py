import json
import os

from .base_config import BaseConfig


class DiceboxConfig(BaseConfig):

    def __init__(self, config_file: str = 'dicebox.config'):
        super().__init__(config_file=config_file)

        ###############################################################################
        # Data Set Options
        ###############################################################################

        # Load user defined config
        if 'DATASET' in os.environ:
            self.DATASET = os.environ['DATASET']

        if 'DICEBOX_COMPLIANT_DATASET' in os.environ:
            if os.environ['DICEBOX_COMPLIANT_DATASET'] == 'True':
                self.DICEBOX_COMPLIANT_DATASET = True
            elif os.environ['DICEBOX_COMPLIANT_DATASET'] == 'False':
                self.DICEBOX_COMPLIANT_DATASET = False
            else:
                raise

        if 'NB_CLASSES' in os.environ:
            self.NB_CLASSES = int(os.environ['NB_CLASSES'])

        if 'IMAGE_WIDTH' in os.environ:
            self.IMAGE_WIDTH = int(os.environ['IMAGE_WIDTH'])

        if 'IMAGE_HEIGHT' in os.environ:
            self.IMAGE_HEIGHT = int(os.environ['IMAGE_HEIGHT'])

        if 'DATA_BASE_DIRECTORY' in os.environ:
            self.DATA_BASE_DIRECTORY = os.environ['DATA_BASE_DIRECTORY']

        ###############################################################################
        # Build Calculated Configs
        ###############################################################################
        self.NETWORK_NAME = "%s_%ix%i" % (self.DATASET, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

        self.INPUT_SHAPE = (self.IMAGE_WIDTH * self.IMAGE_HEIGHT,)

        self.DATA_DIRECTORY = "%s/%s/data/" % (self.DATA_BASE_DIRECTORY, self.NETWORK_NAME)

        ###############################################################################
        # Neural Network Taxonomy Options
        ###############################################################################
        if 'MIN_NEURONS' in os.environ:
            self.MIN_NEURONS = int(os.environ['MIN_NEURONS'])

        if 'MAX_NEURONS' in os.environ:
            self.MAX_NEURONS = int(os.environ['MAX_NEURONS'])

        if 'MIN_LAYERS' in os.environ:
            self.MIN_LAYERS = int(os.environ['MIN_LAYERS'])

        if 'MAX_LAYERS' in os.environ:
            self.MAX_LAYERS = int(os.environ['MAX_LAYERS'])

        if 'LAYER_TYPES' in os.environ:
            self.LAYER_TYPES = os.environ['LAYER_TYPES']

        if 'ACTIVATION' in os.environ:
            self.ACTIVATION = os.environ['ACTIVATION']

        if 'OPTIMIZER' in os.environ:
            self.OPTIMIZER = os.environ['OPTIMIZER']

        self.TAXONOMY = {
            'min_neurons': self.MIN_NEURONS,
            'max_neurons': self.MAX_NEURONS,
            'min_layers': self.MIN_LAYERS,
            'max_layers': self.MAX_LAYERS,
            'layer_types': json.loads(self.LAYER_TYPES),
            'activation': json.loads(self.ACTIVATION),
            'optimizer': json.loads(self.OPTIMIZER)
        }

        ###############################################################################
        # Evolution Options
        ###############################################################################
        if 'EPOCHS' in os.environ:
            self.EPOCHS = int(os.environ['EPOCHS'])

        # Number of times to evolve the population.
        if 'GENERATIONS' in os.environ:
            self.GENERATIONS = int(os.environ['GENERATIONS'])

        # Number of networks in each generation.
        if 'POPULATION' in os.environ:
            self.POPULATION = int(os.environ['POPULATION'])

        if 'NOISE' in os.environ:
            self.NOISE = float(os.environ['NOISE'])

        ###############################################################################
        # Training Options / Settings for the 1920x1080 dataset
        ###############################################################################
        if 'BATCH_SIZE' in os.environ:
            self.BATCH_SIZE = int(os.environ['BATCH_SIZE'])

        if 'TRAIN_BATCH_SIZE' in os.environ:
            self.TRAIN_BATCH_SIZE = int(os.environ['TRAIN_BATCH_SIZE'])

        if 'TEST_BATCH_SIZE' in os.environ:
            self.TEST_BATCH_SIZE = int(os.environ['TEST_BATCH_SIZE'])

        if 'LOAD_BEST_WEIGHTS_ON_START' in os.environ:
            if os.environ['LOAD_BEST_WEIGHTS_ON_START'] == 'False':
                self.LOAD_BEST_WEIGHTS_ON_START = False
            elif os.environ['LOAD_BEST_WEIGHTS_ON_START'] == 'True':
                self.LOAD_BEST_WEIGHTS_ON_START = True
            else:
                raise

        ###############################################################################
        # Direcrtory Options
        ###############################################################################
        if 'LOGS_DIR' in os.environ:
            self.LOGS_DIR = os.environ['LOGS_DIR']

        if 'WEIGHTS_DIR' in os.environ:
            self.WEIGHTS_DIR = os.environ['WEIGHTS_DIR']

        if 'TMP_DIR' in os.environ:
            self.TMP_DIR = os.environ['TMP_DIR']

        ###############################################################################
        # Server Options
        ###############################################################################
        if 'API_ACCESS_KEY' in os.environ:
            self.API_ACCESS_KEY = os.environ['API_ACCESS_KEY']

        if 'API_VERSION' in os.environ:
            self.API_VERSION = os.environ['API_VERSION']

        if 'LISTENING_HOST' in os.environ:
            self.LISTENING_HOST = os.environ['LISTENING_HOST']

        if 'FLASK_DEBUG' in os.environ:
            if os.environ['FLASK_DEBUG'] == 'True':
                self.FLASK_DEBUG = True
            elif os.environ['FLASK_DEBUG'] == 'False':
                self.FLASK_DEBUG = False
            else:
                raise

        if 'MODEL_WEIGHTS_FILENAME' in os.environ:
            self.MODEL_WEIGHTS_FILENAME = os.environ['MODEL_WEIGHTS_FILENAME']

        ###############################################################################
        # Sensory Service Options
        ###############################################################################
        if 'SENSORY_SERVER' in os.environ:
            self.SENSORY_SERVER = os.environ['SENSORY_SERVER']

        if 'SENSORY_PORT' in os.environ:
            self.SENSORY_PORT = int(os.environ['SENSORY_PORT'])

        if 'SENSORY_URI' in os.environ:
            self.SENSORY_URI = os.environ['SENSORY_URI']

        if 'SENSORY_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_EXCHANGE = os.environ['SENSORY_SERVICE_RABBITMQ_EXCHANGE']

        if 'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = os.environ[
                'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY']

        if 'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = os.environ[
                'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE']

        if 'SENSORY_SERVICE_RABBITMQ_URI' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_URI = os.environ['SENSORY_SERVICE_RABBITMQ_URI']

        if 'SENSORY_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_USERNAME = os.environ['SENSORY_SERVICE_RABBITMQ_USERNAME']

        if 'SENSORY_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_PASSWORD = os.environ['SENSORY_SERVICE_RABBITMQ_PASSWORD']

        if 'SENSORY_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_SERVER = os.environ['SENSORY_SERVICE_RABBITMQ_SERVER']

        if 'SENSORY_SERVICE_RABBITMQ_PORT' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_PORT = int(os.environ['SENSORY_SERVICE_RABBITMQ_PORT'])

        if 'SENSORY_SERVICE_RABBITMQ_VHOST' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_VHOST = os.environ['SENSORY_SERVICE_RABBITMQ_VHOST']

        self.SENSORY_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
            self.SENSORY_SERVICE_RABBITMQ_URI,
            self.SENSORY_SERVICE_RABBITMQ_USERNAME,
            self.SENSORY_SERVICE_RABBITMQ_PASSWORD,
            self.SENSORY_SERVICE_RABBITMQ_SERVER,
            self.SENSORY_SERVICE_RABBITMQ_PORT,
            self.SENSORY_SERVICE_RABBITMQ_VHOST
        )

        if 'SENSORY_SERVICE_SHARD_SIZE' in os.environ:
            self.SENSORY_SERVICE_SHARD_SIZE = int(os.environ['SENSORY_SERVICE_SHARD_SIZE'])

        ###############################################################################
        # Training Service Options
        ###############################################################################

        if 'TRAINING_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_EXCHANGE = os.environ['TRAINING_SERVICE_RABBITMQ_EXCHANGE']

        if 'TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = os.environ[
                'TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']

        if 'TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = os.environ[
                'TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']

        if 'TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = os.environ['TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI']

        if 'TRAINING_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_USERNAME = os.environ['TRAINING_SERVICE_RABBITMQ_USERNAME']

        if 'TRAINING_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_PASSWORD = os.environ['TRAINING_SERVICE_RABBITMQ_PASSWORD']

        if 'TRAINING_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_SERVICE_RABBITMQ_SERVER']

        if 'TRAINING_SERVICE_RABBITMQ_PORT' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_PORT = int(os.environ['TRAINING_SERVICE_RABBITMQ_PORT'])

        if 'TRAINING_SERVICE_RABBITMQ_VHOST' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_VHOST = os.environ['TRAINING_SERVICE_RABBITMQ_VHOST']

        self.TRAINING_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
            self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI,
            self.TRAINING_SERVICE_RABBITMQ_USERNAME,
            self.TRAINING_SERVICE_RABBITMQ_PASSWORD,
            self.TRAINING_SERVICE_RABBITMQ_SERVER,
            self.TRAINING_SERVICE_RABBITMQ_PORT,
            self.TRAINING_SERVICE_RABBITMQ_VHOST
        )

        ###############################################################################
        # Training Processor Options
        ###############################################################################

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = os.environ[
                'TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER']

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = int(os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT'])

        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URI,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST
        )

        ###############################################################################
        # Client Options
        ###############################################################################

        if 'CLASSIFICATION_SERVER' in os.environ:
            self.CLASSIFICATION_SERVER = os.environ['CLASSIFICATION_SERVER']

        if 'CLASSIFICATION_SERVER_PORT' in os.environ:
            self.CLASSIFICATION_SERVER_PORT = int(os.environ['CLASSIFICATION_SERVER_PORT'])

        if 'CLASSIFICATION_SERVER_URI' in os.environ:
            self.CLASSIFICATION_SERVER_URI = os.environ['CLASSIFICATION_SERVER_URI']
