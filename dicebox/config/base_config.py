"""Handles loading dicebox.config.

###############################################################################
# Local Config File Handler
# Copyright (c) 2017-2019 Joshua Burt
###############################################################################
"""

###############################################################################
# Dependencies
###############################################################################
import ConfigParser
import json
import urllib
import logging


class BaseConfig(object):

    def __init__(self, config_file='./dicebox.config', lonetar_model_file='./dicebox.lonestar.json'):
        ###############################################################################
        # Create config objects.
        ###############################################################################
        local_config = ConfigParser.ConfigParser()
        local_config.read(config_file)

        # v2 model support
        self.LONESTAR_DICEBOX_MODEL = None
        try:
            model_file = open(lonetar_model_file)
        except IOError as e:
            logging.error(e.message)
            logging.error('Unable to open (%s).  Will not load a lonestar model.', lonetar_model_file)
        else:
            with model_file:
                self.LONESTAR_DICEBOX_MODEL = json.load(model_file)
                # Fix the tuple..
                self.LONESTAR_DICEBOX_MODEL['input_shape'] = [self.LONESTAR_DICEBOX_MODEL['input_shape'], ]


        ###############################################################################
        # Data Set Options
        ###############################################################################
        
        # Load user defined config
        self.DATASET = local_config.get('DATASET', 'name')
        self.DICEBOX_COMPLIANT_DATASET = local_config.getboolean('DATASET', 'dicebox_compliant')
        self.NB_CLASSES = local_config.getint('DATASET', 'categories')
        self.IMAGE_WIDTH = local_config.getint('DATASET', 'image_width')
        self.IMAGE_HEIGHT = local_config.getint('DATASET', 'image_height')
        self.DATA_BASE_DIRECTORY = local_config.get('DIRECTORY', 'dataset_base_directory')

        # Build Calculated Configs
        self.NETWORK_NAME = "%s_%ix%i" % (self.DATASET, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)
        self.INPUT_SHAPE = (self.IMAGE_WIDTH * self.IMAGE_HEIGHT,)
        self.DATA_DIRECTORY = "%s/%s/data/" % (self.DATA_BASE_DIRECTORY, self.NETWORK_NAME)


        ###############################################################################
        # Neural Network Taxonomy Options
        ###############################################################################
        self.NB_NEURONS = local_config.get('TAXONOMY', 'neurons')
        self.NB_LAYERS = local_config.get('TAXONOMY', 'layers')
        self.ACTIVATION = local_config.get('TAXONOMY', 'activation')
        self.OPTIMIZER = local_config.get('TAXONOMY', 'optimizer')

        self.NN_PARAM_CHOICES = {
            'nb_neurons': json.loads(self.NB_NEURONS),
            'nb_layers': json.loads(self.NB_LAYERS),
            'activation': json.loads(self.ACTIVATION),
            'optimizer': json.loads(self.OPTIMIZER)
        }


        ###############################################################################
        # Lonestar Options
        ###############################################################################
        self.NB_LONESTAR_NEURONS = local_config.getint('LONESTAR', 'neurons')
        self.NB_LONESTAR_LAYERS = local_config.getint('LONESTAR', 'layers')
        self.LONESTAR_ACTIVATION = local_config.get('LONESTAR', 'activation')
        self.LONESTAR_OPTIMIZER = local_config.get('LONESTAR', 'optimizer')

        self.NN_LONESTAR_PARAMS = {
            'nb_neurons': self.NB_LONESTAR_NEURONS,
            'nb_layers': self.NB_LONESTAR_LAYERS,
            'activation': self.LONESTAR_ACTIVATION,
            'optimizer': self.LONESTAR_OPTIMIZER
        }


        ###############################################################################
        # Evolution Options
        ###############################################################################
        self.EPOCHS = local_config.getint('EVOLUTION', 'epochs')

        # Number of times to evole the population.
        self.GENERATIONS = local_config.getint('EVOLUTION', 'generations')

        # Number of networks in each generation.
        self.POPULATION = local_config.getint('EVOLUTION', 'population')
        self.NOISE = local_config.getfloat('GLOBAL', 'noise')


        ###############################################################################
        # Training Options / Settings for the 1920x1080 dataset
        ###############################################################################
        self.BATCH_SIZE = local_config.getint('TRAINING', 'batch_size')
        self.TRAIN_BATCH_SIZE = local_config.getint('TRAINING', 'train_batch_size')
        self.TEST_BATCH_SIZE = local_config.getint('TRAINING', 'test_batch_size')
        self.LOAD_BEST_WEIGHTS_ON_START = local_config.getboolean('TRAINING', 'load_best_weights_on_start')


        ###############################################################################
        # Direcrtory Options
        ###############################################################################
        self.LOGS_DIR = local_config.get('DIRECTORY', 'logs_dir')
        self.WEIGHTS_DIR = local_config.get('DIRECTORY', 'weights_dir')
        self.TMP_DIR = local_config.get('DIRECTORY', 'tmp_dir')


        ###############################################################################
        # Server Options
        ###############################################################################
        self.API_ACCESS_KEY = local_config.get('SERVER', 'api_access_key')
        self.API_VERSION = local_config.get('SERVER', 'api_version')
        self.LISTENING_HOST = local_config.get('SERVER', 'listening_host')
        self.FLASK_DEBUG = local_config.getboolean('SERVER', 'flask_debug')
        self.MODEL_WEIGHTS_FILENAME = local_config.get('SERVER', 'model_weights_filename')


        ###############################################################################
        # Sensory Service Options
        ###############################################################################
        self.SENSORY_SERVER = local_config.get('SENSORY_SERVICE', 'sensory_server')
        self.SENSORY_PORT = local_config.getint('SENSORY_SERVICE', 'sensory_port')
        self.SENSORY_URI = local_config.get('SENSORY_SERVICE', 'sensory_uri')

        self.SENSORY_SERVICE_RABBITMQ_EXCHANGE = local_config.get('SENSORY_SERVICE', 'rabbitmq_exchange')
        self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = local_config.get(
            'SENSORY_SERVICE', 'rabbitmq_batch_request_routing_key')
        self.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = local_config.get(
            'SENSORY_SERVICE', 'rabbitmq_batch_request_task_queue')

        self.SENSORY_SERVICE_RABBITMQ_URI = local_config.get('SENSORY_SERVICE', 'rabbitmq_uri')
        self.SENSORY_SERVICE_RABBITMQ_USERNAME = local_config.get('SENSORY_SERVICE', 'rabbitmq_username')
        self.SENSORY_SERVICE_RABBITMQ_PASSWORD = local_config.get('SENSORY_SERVICE', 'rabbitmq_password')
        self.SENSORY_SERVICE_RABBITMQ_SERVER = local_config.get('SENSORY_SERVICE', 'rabbitmq_server')
        self.SENSORY_SERVICE_RABBITMQ_PORT = local_config.get('SENSORY_SERVICE', 'rabbitmq_port')
        self.SENSORY_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
            local_config.get('SENSORY_SERVICE', 'rabbitmq_vhost'))

        self.SENSORY_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
            self.SENSORY_SERVICE_RABBITMQ_URI,
            self.SENSORY_SERVICE_RABBITMQ_USERNAME,
            self.SENSORY_SERVICE_RABBITMQ_PASSWORD,
            self.SENSORY_SERVICE_RABBITMQ_SERVER,
            self.SENSORY_SERVICE_RABBITMQ_PORT,
            self.SENSORY_SERVICE_RABBITMQ_VHOST
        )
        self.SENSORY_SERVICE_SHARD_SIZE = local_config.getint('SENSORY_SERVICE', 'shard_size')


        ###############################################################################
        # Training Service Options
        ###############################################################################
        self.TRAINING_SERVICE_RABBITMQ_EXCHANGE = local_config.get(
            'TRAINING_SERVICE', 'rabbitmq_exchange')
        self.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = local_config.get(
            'TRAINING_SERVICE', 'rabbitmq_batch_request_routing_key')
        self.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = local_config.get(
            'TRAINING_SERVICE', 'rabbitmq_train_request_task_queue')
        self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST = local_config.get('TRAINING_SERVICE', 'rabbitmq_vhost')
        self.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = local_config.get('TRAINING_SERVICE', 'rabbitmq_uri')
        self.TRAINING_SERVICE_RABBITMQ_USERNAME = local_config.get('TRAINING_SERVICE', 'rabbitmq_username')
        self.TRAINING_SERVICE_RABBITMQ_PASSWORD = local_config.get('TRAINING_SERVICE', 'rabbitmq_password')
        self.TRAINING_SERVICE_RABBITMQ_SERVER = local_config.get('TRAINING_SERVICE', 'rabbitmq_server')
        self.TRAINING_SERVICE_RABBITMQ_PORT = local_config.get('TRAINING_SERVICE', 'rabbitmq_port')
        self.TRAINING_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
            local_config.get('TRAINING_SERVICE', 'rabbitmq_vhost'))
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
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_exchange')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_batch_request_routing_key')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_train_request_task_queue')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_vhost')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_uri')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_username')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_password')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_server')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = local_config.get(
            'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_port')
        self.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
            local_config.get('TRAINING_PROCESSOR_SERVICE', 'rabbitmq_vhost'))
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
        self.CLASSIFICATION_SERVER = local_config.get('CLIENT', 'classification_server')
        self.SERVER_PORT = local_config.getint('CLIENT', 'classification_port')
        self.SERVER_URI = local_config.get('CLIENT', 'classification_uri')
