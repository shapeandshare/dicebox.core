"""Allow over-riding the defaults with non-secure ENV variables, or secure docker secrets

###############################################################################
# Local Config File Handler
# Copyright (c) 2017-2019 Joshua Burt
###############################################################################
"""

###############################################################################
# Dependencies
###############################################################################
import os
import json
from dicebox.config.base_config import BaseConfig


class DiceboxConfig(object):
    
    def __init__(self, config_file='./dicebox.config', lonestar_model_file='./dicebox.lonestar.json'):
        self.dc = BaseConfig(config_file=config_file, lonestar_model_file=lonestar_model_file)
    
        
        ###############################################################################
        # Data Set Options
        ###############################################################################
        
        # Load user defined config
        self.DATASET = self.dc.DATASET
        if 'DATASET' in os.environ:
            self.DATASET = os.environ['DATASET']
        
        self.DICEBOX_COMPLIANT_DATASET = self.dc.DICEBOX_COMPLIANT_DATASET
        if 'DICEBOX_COMPLIANT_DATASET' in os.environ:
            self.DICEBOX_COMPLIANT_DATASET = os.environ['DICEBOX_COMPLIANT_DATASET']

        self.NB_CLASSES = self.dc.NB_CLASSES
        if 'NB_CLASSES' in os.environ:
            self.NB_CLASSES = int(os.environ['NB_CLASSES'])

        self.IMAGE_WIDTH = self.dc.IMAGE_WIDTH
        if 'IMAGE_WIDTH' in os.environ:
            self.IMAGE_WIDTH = int(os.environ['IMAGE_WIDTH'])

        self.IMAGE_HEIGHT = self.dc.IMAGE_HEIGHT
        if 'IMAGE_HEIGHT' in os.environ:
            self.IMAGE_HEIGHT = int(os.environ['IMAGE_HEIGHT'])

        self.DATA_BASE_DIRECTORY = self.dc.DATA_BASE_DIRECTORY
        if 'DATA_BASE_DIRECTORY' in os.environ:
            self.DATA_BASE_DIRECTORY = os.environ['DATA_BASE_DIRECTORY']


        ###############################################################################
        # Build Calculated Configs
        ###############################################################################
        self.NETWORK_NAME = "%s_%ix%i" % (self.DATASET, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        if 'NETWORK_NAME' in os.environ:
            self.NETWORK_NAME = os.environ['NETWORK_NAME']

        self.INPUT_SHAPE = (self.IMAGE_WIDTH * self.IMAGE_HEIGHT,)
        if 'INPUT_SHAPE' in os.environ:
            self.INPUT_SHAPE = os.environ['INPUT_SHAPE']

        self.DATA_DIRECTORY = "%s/%s/data/" % (self.DATA_BASE_DIRECTORY, self.NETWORK_NAME)
        if 'DATA_DIRECTORY' in os.environ:
            self.DATA_DIRECTORY = os.environ['DATA_DIRECTORY']


        ###############################################################################
        # Neural Network Taxonomy Options
        ###############################################################################
        self.NB_NEURONS = self.dc.NB_NEURONS
        if 'NB_NEURONS' in os.environ:
            self.NB_NEURONS = os.environ['NB_NEURONS']

        self.NB_LAYERS = self.dc.NB_LAYERS
        if 'NB_LAYERS' in os.environ:
            self.NB_LAYERS = os.environ['NB_LAYERS']

        self.ACTIVATION = self.dc.ACTIVATION
        if 'ACTIVATION' in os.environ:
            self.ACTIVATION = os.environ['ACTIVATION']

        self.OPTIMIZER = self.dc.OPTIMIZER
        if 'OPTIMIZER' in os.environ:
            self.OPTIMIZER = os.environ['OPTIMIZER']

        self.NN_PARAM_CHOICES = {
            'nb_neurons': json.loads(self.NB_NEURONS),
            'nb_layers': json.loads(self.NB_LAYERS),
            'activation': json.loads(self.ACTIVATION),
            'optimizer': json.loads(self.OPTIMIZER)
        }

        self.MIN_NEURONS = self.dc.MIN_NEURONS
        self.MAX_NEURONS = self.dc.MAX_NEURONS
        self.MIN_LAYERS = self.dc.MIN_LAYERS
        self.MAX_LAYERS = self.dc.MAX_LAYERS
        self.LAYER_TYPES = self.dc.LAYER_TYPES

        self.TAXONOMY = {
            'min_neurons': self.MIN_NEURONS,
            'max_neurons': self.MAX_NEURONS,
            'min_layers': self.MIN_LAYERS,
            'max_layers': self.MAX_NEURONS,
            'layer_types': json.loads(self.LAYER_TYPES),
            'activation': json.loads(self.ACTIVATION),
            'optimizer': json.loads(self.OPTIMIZER)
        }


        ###############################################################################
        # Lonestar Options
        ###############################################################################
        self.NB_LONESTAR_NEURONS = self.dc.NB_LONESTAR_NEURONS
        if 'NB_LONESTAR_NEURONS' in os.environ:
            self.NB_LONESTAR_NEURONS = int(os.environ['NB_LONESTAR_NEURONS'])

        self.NB_LONESTAR_LAYERS = self.dc.NB_LONESTAR_LAYERS
        if 'NB_LONESTAR_LAYERS' in os.environ:
            self.NB_LONESTAR_LAYERS = int(os.environ['NB_LONESTAR_LAYERS'])

        self.LONESTAR_ACTIVATION = self.dc.LONESTAR_ACTIVATION
        if 'LONESTAR_ACTIVATION' in os.environ:
            self.LONESTAR_ACTIVATION = os.environ['LONESTAR_ACTIVATION']

        self.LONESTAR_OPTIMIZER = self.dc.LONESTAR_OPTIMIZER
        if 'LONESTAR_OPTIMIZER' in os.environ:
            self.LONESTAR_OPTIMIZER = os.environ['LONESTAR_OPTIMIZER']

        self.NN_LONESTAR_PARAMS = {
            'nb_neurons': self.NB_LONESTAR_NEURONS,
            'nb_layers': self.NB_LONESTAR_LAYERS,
            'activation': self.LONESTAR_ACTIVATION,
            'optimizer': self.LONESTAR_OPTIMIZER
        }

        # support for v2 model
        self.LONESTAR_DICEBOX_MODEL = self.dc.LONESTAR_DICEBOX_MODEL
        if 'LONESTAR_DICEBOX_MODEL' in os.environ:
            self.LONESTAR_DICEBOX_MODEL = json.loads(os.environ['LONESTAR_DICEBOX_MODEL'])


        ###############################################################################
        # Evolution Options
        ###############################################################################
        self.EPOCHS = self.dc.EPOCHS
        if 'EPOCHS' in os.environ:
            self.EPOCHS = int(os.environ['EPOCHS'])

        self.GENERATIONS = self.dc.GENERATIONS  # Number of times to evole the population.
        if 'GENERATIONS' in os.environ:
            self.GENERATIONS = int(os.environ['GENERATIONS'])

        self.POPULATION = self.dc.POPULATION  # Number of networks in each generation.
        if 'POPULATION' in os.environ:
            self.POPULATION = int(os.environ['POPULATION'])

        self.NOISE = self.dc.NOISE
        if 'NOISE' in os.environ:
            self.NOISE = float(os.environ['NOISE'])


        ###############################################################################
        # Training Options / Settings for the 1920x1080 dataset
        ###############################################################################
        self.BATCH_SIZE = self.dc.BATCH_SIZE
        if 'BATCH_SIZE' in os.environ:
            self.BATCH_SIZE = int(os.environ['BATCH_SIZE'])

        self.TRAIN_BATCH_SIZE = self.dc.TRAIN_BATCH_SIZE
        if 'TRAIN_BATCH_SIZE' in os.environ:
            self.TRAIN_BATCH_SIZE = int(os.environ['TRAIN_BATCH_SIZE'])

        self.TEST_BATCH_SIZE = self.dc.TEST_BATCH_SIZE
        if 'TEST_BATCH_SIZE' in os.environ:
            self.TEST_BATCH_SIZE = int(os.environ['TEST_BATCH_SIZE'])

        self.LOAD_BEST_WEIGHTS_ON_START = self.dc.LOAD_BEST_WEIGHTS_ON_START
        if 'LOAD_BEST_WEIGHTS_ON_START' in os.environ:
            if os.environ['LOAD_BEST_WEIGHTS_ON_START'] == 'False':
                self.LOAD_BEST_WEIGHTS_ON_START = False


        ###############################################################################
        # Direcrtory Options
        ###############################################################################
        self.LOGS_DIR = self.dc.LOGS_DIR
        if 'LOGS_DIR' in os.environ:
            self.LOGS_DIR = os.environ['LOGS_DIR']

        self.WEIGHTS_DIR = self.dc.WEIGHTS_DIR
        if 'WEIGHTS_DIR' in os.environ:
            self.WEIGHTS_DIR = os.environ['WEIGHTS_DIR']

        self.TMP_DIR = self.dc.TMP_DIR
        if 'TMP_DIR' in os.environ:
            self.MP_DIR = os.environ['TMP_DIR']


        ###############################################################################
        # Server Options
        ###############################################################################
        self.API_ACCESS_KEY = self.dc.API_ACCESS_KEY
        if 'API_ACCESS_KEY' in os.environ:
            self.API_ACCESS_KEY = os.environ['API_ACCESS_KEY']

        self.API_VERSION = self.dc.API_VERSION
        if 'API_VERSION' in os.environ:
            self.API_VERSION = os.environ['API_VERSION']

        self.LISTENING_HOST = self.dc.LISTENING_HOST
        if 'LISTENING_HOST' in os.environ:
            self.LISTENING_HOST = os.environ['LISTENING_HOST']

        self.FLASK_DEBUG = self.dc.FLASK_DEBUG
        if 'FLASK_DEBUG' in os.environ:
            if os.environ['FLASK_DEBUG'] == 'True':
                self.FLASK_DEBUG = True

        self.MODEL_WEIGHTS_FILENAME = self.dc.MODEL_WEIGHTS_FILENAME
        if 'MODEL_WEIGHTS_FILENAME' in os.environ:
            self.MODEL_WEIGHTS_FILENAME = os.environ['MODEL_WEIGHTS_FILENAME']


        ###############################################################################
        # Sensory Service Options
        ###############################################################################
        self.SENSORY_SERVER = self.dc.SENSORY_SERVER
        if 'SENSORY_SERVER' in os.environ:
            self.SENSORY_SERVER = os.environ['SENSORY_SERVER']

        self.SENSORY_PORT = self.dc.SENSORY_PORT
        if 'SENSORY_PORT' in os.environ:
            self.SENSORY_PORT = os.environ['SENSORY_PORT']

        self.SENSORY_URI = self.dc.SENSORY_URI
        if 'SENSORY_URI' in os.environ:
            self.SENSORY_URI = os.environ['SENSORY_URI']

        self.SENSORY_SERVICE_RABBITMQ_EXCHANGE = self.dc.SENSORY_SERVICE_RABBITMQ_EXCHANGE
        if 'SENSORY_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_EXCHANGE = os.environ['SENSORY_SERVICE_RABBITMQ_EXCHANGE']

        self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY
        if 'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = os.environ['SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY']

        self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = self.dc.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE
        if 'SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = os.environ['SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE']

        self.SENSORY_SERVICE_RABBITMQ_URI = self.dc.SENSORY_SERVICE_RABBITMQ_URI
        if 'SENSORY_SERVICE_RABBITMQ_URI' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_URI = os.environ['SENSORY_SERVICE_RABBITMQ_URI']

        self.SENSORY_SERVICE_RABBITMQ_USERNAME = self.dc.SENSORY_SERVICE_RABBITMQ_USERNAME
        if 'SENSORY_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_USERNAME = os.environ['SENSORY_SERVICE_RABBITMQ_USERNAME']

        self.SENSORY_SERVICE_RABBITMQ_PASSWORD = self.dc.SENSORY_SERVICE_RABBITMQ_PASSWORD
        if 'SENSORY_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_PASSWORD = os.environ['SENSORY_SERVICE_RABBITMQ_PASSWORD']

        self.SENSORY_SERVICE_RABBITMQ_SERVER = self.dc.SENSORY_SERVICE_RABBITMQ_SERVER
        if 'SENSORY_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_SERVER = os.environ['SENSORY_SERVICE_RABBITMQ_SERVER']

        self.SENSORY_SERVICE_RABBITMQ_PORT = self.dc.SENSORY_SERVICE_RABBITMQ_PORT
        if 'SENSORY_SERVICE_RABBITMQ_PORT' in os.environ:
            self.SENSORY_SERVICE_RABBITMQ_PORT = os.environ['SENSORY_SERVICE_RABBITMQ_PORT']

        self.SENSORY_SERVICE_RABBITMQ_VHOST = self.dc.SENSORY_SERVICE_RABBITMQ_VHOST
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
        self.SENSORY_SERVICE_SHARD_SIZE = self.dc.SENSORY_SERVICE_SHARD_SIZE
        if 'SENSORY_SERVICE_SHARD_SIZE' in os.environ:
            self.SENSORY_SERVICE_SHARD_SIZE = int(os.environ['SENSORY_SERVICE_SHARD_SIZE'])



        ###############################################################################
        # Training Service Options
        ###############################################################################
        self.TRAINING_SERVICE_RABBITMQ_EXCHANGE = self.dc.TRAINING_SERVICE_RABBITMQ_EXCHANGE
        if 'TRAINING_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_EXCHANGE = os.environ['TRAINING_SERVICE_RABBITMQ_EXCHANGE']

        self.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = self.dc.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY
        if 'TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = os.environ['TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']

        self.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = self.dc.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE
        if 'TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = os.environ['TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']

        self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST = self.dc.TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST
        if 'TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST = os.environ['TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST']

        self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = self.dc.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI
        if 'TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = os.environ['TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI']

        self.TRAINING_SERVICE_RABBITMQ_USERNAME = self.dc.TRAINING_SERVICE_RABBITMQ_USERNAME
        if 'TRAINING_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_USERNAME = os.environ['TRAINING_SERVICE_RABBITMQ_USERNAME']

        self.TRAINING_SERVICE_RABBITMQ_PASSWORD = self.dc.TRAINING_SERVICE_RABBITMQ_PASSWORD
        if 'TRAINING_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_PASSWORD = os.environ['TRAINING_SERVICE_RABBITMQ_PASSWORD']

        self.TRAINING_SERVICE_RABBITMQ_SERVER = self.dc.TRAINING_SERVICE_RABBITMQ_SERVER
        if 'TRAINING_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_SERVICE_RABBITMQ_SERVER']

        self.TRAINING_SERVICE_RABBITMQ_PORT = self.dc.TRAINING_SERVICE_RABBITMQ_PORT
        if 'TRAINING_SERVICE_RABBITMQ_PORT' in os.environ:
            self.TRAINING_SERVICE_RABBITMQ_PORT = os.environ['TRAINING_SERVICE_RABBITMQ_PORT']

        self.TRAINING_SERVICE_RABBITMQ_VHOST = self.dc.TRAINING_SERVICE_RABBITMQ_VHOST
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
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = self.dc.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST
        if 'TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST' in os.environ:
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST']

        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT,
            self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST
        )


        ###############################################################################
        # Client Options
        ###############################################################################
        self.CLASSIFICATION_SERVER = self.dc.CLASSIFICATION_SERVER
        if 'CLASSIFICATION_SERVER' in os.environ:
            self.CLASSIFICATION_SERVER = os.environ['CLASSIFICATION_SERVER']

        self.SERVER_PORT = self.dc.SERVER_PORT
        if 'SERVER_PORT' in os.environ:
            self.SERVER_PORT = int(os.environ['SERVER_PORT'])

        self.SERVER_URI = self.dc.SERVER_URI
        if 'SERVER_URI' in os.environ:
            self.SERVER_URI = os.environ['SERVER_URI']
