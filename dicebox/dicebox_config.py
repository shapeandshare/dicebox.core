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


###############################################################################
# Create config objects.
###############################################################################
LOCAL_CONIFG = ConfigParser.ConfigParser()
LOCAL_CONIFG.read('./dicebox.config')


###############################################################################
# Data Set Options
###############################################################################

# Load user defined config
DATASET = LOCAL_CONIFG.get('DATASET', 'name')
DICEBOX_COMPLIANT_DATASET = LOCAL_CONIFG.getboolean('DATASET', 'dicebox_compliant')
NB_CLASSES = LOCAL_CONIFG.getint('DATASET', 'categories')
IMAGE_WIDTH = LOCAL_CONIFG.getint('DATASET', 'image_width')
IMAGE_HEIGHT = LOCAL_CONIFG.getint('DATASET', 'image_height')
DATA_BASE_DIRECTORY = LOCAL_CONIFG.get('DIRECTORY', 'dataset_base_directory')

# Build Calculated Configs
NETWORK_NAME = "%s_%ix%i" % (DATASET, IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH*IMAGE_HEIGHT,)
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)


###############################################################################
# Neural Network Taxonomy Options
###############################################################################
NB_NEURONS = LOCAL_CONIFG.get('TAXONOMY', 'neurons')
NB_LAYERS = LOCAL_CONIFG.get('TAXONOMY', 'layers')
ACTIVATION = LOCAL_CONIFG.get('TAXONOMY', 'activation')
OPTIMIZER = LOCAL_CONIFG.get('TAXONOMY', 'optimizer')

NN_PARAM_CHOICES = {
    'nb_neurons': json.loads(NB_NEURONS),
    'nb_layers': json.loads(NB_LAYERS),
    'activation': json.loads(ACTIVATION),
    'optimizer': json.loads(OPTIMIZER)
}


###############################################################################
# Lonestar Options
# {'nb_layers': 2, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 987}
# 07/03/2017 04:02:49 AM - INFO - Network accuracy: 97.27%
###############################################################################
NB_LONESTAR_NEURONS = LOCAL_CONIFG.getint('LONESTAR', 'neurons')
NB_LONESTAR_LAYERS = LOCAL_CONIFG.getint('LONESTAR', 'layers')
LONESTAR_ACTIVATION = LOCAL_CONIFG.get('LONESTAR', 'activation')
LONESTAR_OPTIMIZER = LOCAL_CONIFG.get('LONESTAR', 'optimizer')

NN_LONESTAR_PARAMS = {
    'nb_neurons': NB_LONESTAR_NEURONS,
    'nb_layers': NB_LONESTAR_LAYERS,
    'activation': LONESTAR_ACTIVATION,
    'optimizer': LONESTAR_OPTIMIZER
}


###############################################################################
# Evolution Options
###############################################################################
EPOCHS = LOCAL_CONIFG.getint('EVOLUTION', 'epochs')

# Number of times to evole the population.
GENERATIONS = LOCAL_CONIFG.getint('EVOLUTION', 'generations')

# Number of networks in each generation.
POPULATION = LOCAL_CONIFG.getint('EVOLUTION', 'population')
NOISE = LOCAL_CONIFG.getfloat('GLOBAL', 'noise')


###############################################################################
# Training Options / Settings for the 1920x1080 dataset
###############################################################################
BATCH_SIZE = LOCAL_CONIFG.getint('TRAINING', 'batch_size')
TRAIN_BATCH_SIZE = LOCAL_CONIFG.getint('TRAINING', 'train_batch_size')
TEST_BATCH_SIZE = LOCAL_CONIFG.getint('TRAINING', 'test_batch_size')
LOAD_BEST_WEIGHTS_ON_START = LOCAL_CONIFG.getboolean('TRAINING', 'load_best_weights_on_start')

###############################################################################
# Direcrtory Options
###############################################################################
LOGS_DIR = LOCAL_CONIFG.get('DIRECTORY', 'logs_dir')
WEIGHTS_DIR = LOCAL_CONIFG.get('DIRECTORY', 'weights_dir')
TMP_DIR = LOCAL_CONIFG.get('DIRECTORY', 'tmp_dir')


###############################################################################
# Server Options
###############################################################################
API_ACCESS_KEY = LOCAL_CONIFG.get('SERVER', 'api_access_key')
API_VERSION = LOCAL_CONIFG.get('SERVER', 'api_version')
LISTENING_HOST = LOCAL_CONIFG.get('SERVER', 'listening_host')
FLASK_DEBUG = LOCAL_CONIFG.getboolean('SERVER', 'flask_debug')
MODEL_WEIGHTS_FILENAME = LOCAL_CONIFG.get('SERVER', 'model_weights_filename')


###############################################################################
# Sensory Service Options
###############################################################################
SENSORY_SERVER = LOCAL_CONIFG.get('SENSORY_SERVICE', 'sensory_server')
SENSORY_PORT = LOCAL_CONIFG.getint('SENSORY_SERVICE', 'sensory_port')
SENSORY_URI = LOCAL_CONIFG.get('SENSORY_SERVICE', 'sensory_uri')

SENSORY_SERVICE_RABBITMQ_EXCHANGE = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_exchange')
SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = LOCAL_CONIFG.get(
    'SENSORY_SERVICE', 'rabbitmq_batch_request_routing_key')
SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = LOCAL_CONIFG.get(
    'SENSORY_SERVICE', 'rabbitmq_batch_request_task_queue')

SENSORY_SERVICE_RABBITMQ_URI = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_uri')
SENSORY_SERVICE_RABBITMQ_USERNAME = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_username')
SENSORY_SERVICE_RABBITMQ_PASSWORD = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_password')
SENSORY_SERVICE_RABBITMQ_SERVER = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_server')
SENSORY_SERVICE_RABBITMQ_PORT = LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_port')
SENSORY_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
    LOCAL_CONIFG.get('SENSORY_SERVICE', 'rabbitmq_vhost'))

SENSORY_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
    SENSORY_SERVICE_RABBITMQ_URI,
    SENSORY_SERVICE_RABBITMQ_USERNAME,
    SENSORY_SERVICE_RABBITMQ_PASSWORD,
    SENSORY_SERVICE_RABBITMQ_SERVER,
    SENSORY_SERVICE_RABBITMQ_PORT,
    SENSORY_SERVICE_RABBITMQ_VHOST
)
SENSORY_SERVICE_SHARD_SIZE = LOCAL_CONIFG.getint('SENSORY_SERVICE', 'shard_size')


###############################################################################
# Training Service Options
###############################################################################
TRAINING_SERVICE_RABBITMQ_EXCHANGE = LOCAL_CONIFG.get(
    'TRAINING_SERVICE', 'rabbitmq_exchange')
TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = LOCAL_CONIFG.get(
    'TRAINING_SERVICE', 'rabbitmq_batch_request_routing_key')
TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = LOCAL_CONIFG.get(
    'TRAINING_SERVICE', 'rabbitmq_train_request_task_queue')
TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_vhost')
TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_uri')
TRAINING_SERVICE_RABBITMQ_USERNAME = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_username')
TRAINING_SERVICE_RABBITMQ_PASSWORD = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_password')
TRAINING_SERVICE_RABBITMQ_SERVER = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_server')
TRAINING_SERVICE_RABBITMQ_PORT = LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_port')
TRAINING_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
    LOCAL_CONIFG.get('TRAINING_SERVICE', 'rabbitmq_vhost'))
TRAINING_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
    TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI,
    TRAINING_SERVICE_RABBITMQ_USERNAME,
    TRAINING_SERVICE_RABBITMQ_PASSWORD,
    TRAINING_SERVICE_RABBITMQ_SERVER,
    TRAINING_SERVICE_RABBITMQ_PORT,
    TRAINING_SERVICE_RABBITMQ_VHOST
)


###############################################################################
# Training Processor Options
###############################################################################
TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_exchange')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_batch_request_routing_key')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_train_request_task_queue')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_vhost')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_uri')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_username')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_password')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_server')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = LOCAL_CONIFG.get(
    'TRAINING_PROCESSOR_SERVICE', 'rabbitmq_port')
TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = urllib.quote_plus(
    LOCAL_CONIFG.get('TRAINING_PROCESSOR_SERVICE', 'rabbitmq_vhost'))
TRAINING_PROCESSOR_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI,
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME,
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD,
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER,
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT,
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST
)


###############################################################################
# Client Options
###############################################################################
CLASSIFICATION_SERVER = LOCAL_CONIFG.get('CLIENT', 'classification_server')
SERVER_PORT = LOCAL_CONIFG.getint('CLIENT', 'classification_port')
SERVER_URI = LOCAL_CONIFG.get('CLIENT', 'classification_uri')
