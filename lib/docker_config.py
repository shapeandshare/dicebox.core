# Allow over-riding the defaults with non-secure ENV varaibles, or secure docker secrets

import dicebox_config as default_config
import os

###############################################################################
# Data Set Options
###############################################################################

# Load user defined config
DATASET = default_config.DATASET
if os.environ['DATASET'] is not None:
    DATASET = os.environ['DATASET']

NB_CLASSES = default_config.NB_CLASSES
IMAGE_WIDTH = default_config.IMAGE_WIDTH
IMAGE_HEIGHT = default_config.IMAGE_HEIGHT
DATA_BASE_DIRECTORY = default_config.DATA_BASE_DIRECTORY

# Build Calculated Configs
NETWORK_NAME = default_config.NETWORK_NAME
INPUT_SHAPE = default_config.INPUT_SHAPE
DATA_DIRECTORY = default_config.DATA_DIRECTORY


###############################################################################
# Neural Network Taxonomy Options
###############################################################################
NN_PARAM_CHOICES = default_config.NN_PARAM_CHOICES


###############################################################################
# Lonestar Options
# {'nb_layers': 2, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 987}
# 07/03/2017 04:02:49 AM - INFO - Network accuracy: 97.27%
###############################################################################
NN_LONESTAR_PARAMS = default_config.NN_LONESTAR_PARAMS


###############################################################################
# Evolution Options
###############################################################################
EPOCHS = default_config.EPOCHS
GENERATIONS = default_config.GENERATIONS  # Number of times to evole the population.
POPULATION = default_config.POPULATION  # Number of networks in each generation.
NOISE = default_config.NOISE


###############################################################################
# Training Options / Settings for the 1920x1080 dataset
###############################################################################
BATCH_SIZE = default_config.BATCH_SIZE
TRAIN_BATCH_SIZE = default_config.TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = default_config.TEST_BATCH_SIZE
LOAD_BEST_WEIGHTS_ON_START = default_config.LOAD_BEST_WEIGHTS_ON_START

###############################################################################
# Direcrtory Options
###############################################################################
LOGS_DIR = default_config.LOGS_DIR
WEIGHTS_DIR = default_config.WEIGHTS_DIR
TMP_DIR = default_config.TMP_DIR


###############################################################################
# Server Options
###############################################################################
API_ACCESS_KEY = default_config.API_ACCESS_KEY
API_VERSION = default_config.API_VERSION
LISTENING_HOST = default_config.LISTENING_HOST
FLASK_DEBUG = default_config.FLASK_DEBUG
MODEL_WEIGHTS_FILENAME = default_config.MODEL_WEIGHTS_FILENAME


###############################################################################
# Sensory Service Options
###############################################################################
SENSORY_SERVER = default_config.SENSORY_SERVER
SENSORY_PORT = default_config.SENSORY_PORT
SENSORY_URI = default_config.SENSORY_URI

SENSORY_SERVICE_RABBITMQ_EXCHANGE = default_config.SENSORY_SERVICE_RABBITMQ_EXCHANGE
SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY = default_config.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_ROUTING_KEY
SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE = default_config.SENSORY_SERVICE_RABBITMQ_BATCH_REQUEST_TASK_QUEUE

SENSORY_SERVICE_RABBITMQ_URI = default_config.SENSORY_SERVICE_RABBITMQ_URI

SENSORY_SERVICE_RABBITMQ_USERNAME = default_config.SENSORY_SERVICE_RABBITMQ_USERNAME
if os.environ['SENSORY_SERVICE_RABBITMQ_USERNAME'] is not None:
    SENSORY_SERVICE_RABBITMQ_USERNAME = os.environ['SENSORY_SERVICE_RABBITMQ_USERNAME']

SENSORY_SERVICE_RABBITMQ_PASSWORD = default_config.SENSORY_SERVICE_RABBITMQ_PASSWORD
if os.environ['SENSORY_SERVICE_RABBITMQ_PASSWORD'] is not None:
    SENSORY_SERVICE_RABBITMQ_PASSWORD = os.environ['SENSORY_SERVICE_RABBITMQ_PASSWORD']

SENSORY_SERVICE_RABBITMQ_SERVER = default_config.SENSORY_SERVICE_RABBITMQ_SERVER
if os.environ['SENSORY_SERVICE_RABBITMQ_SERVER'] is not None:
    SENSORY_SERVICE_RABBITMQ_SERVER = os.environ['SENSORY_SERVICE_RABBITMQ_SERVER']

SENSORY_SERVICE_RABBITMQ_PORT = default_config.SENSORY_SERVICE_RABBITMQ_PORT
SENSORY_SERVICE_RABBITMQ_VHOST = default_config.SENSORY_SERVICE_RABBITMQ_VHOST
SENSORY_SERVICE_RABBITMQ_URL = "%s%s:%s@%s:%s/%s" % (
    SENSORY_SERVICE_RABBITMQ_URI,
    SENSORY_SERVICE_RABBITMQ_USERNAME,
    SENSORY_SERVICE_RABBITMQ_PASSWORD,
    SENSORY_SERVICE_RABBITMQ_SERVER,
    SENSORY_SERVICE_RABBITMQ_PORT,
    SENSORY_SERVICE_RABBITMQ_VHOST
)
SENSORY_SERVICE_SHARD_SIZE = default_config.SENSORY_SERVICE_SHARD_SIZE


###############################################################################
# Training Service Options
###############################################################################
TRAINING_SERVICE_RABBITMQ_EXCHANGE = default_config.TRAINING_SERVICE_RABBITMQ_EXCHANGE
TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = default_config.TRAINING_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY
TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = default_config.TRAINING_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE
TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST = default_config.TRAINING_SERVICE_RABBITMQ_RABBITMQ_VHOST
TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI = default_config.TRAINING_SERVICE_RABBITMQ_RABBITMQ_URI

TRAINING_SERVICE_RABBITMQ_USERNAME = default_config.TRAINING_SERVICE_RABBITMQ_USERNAME
if os.environ['TRAINING_SERVICE_RABBITMQ_USERNAME'] is not None:
    TRAINING_SERVICE_RABBITMQ_USERNAME = os.environ['TRAINING_SERVICE_RABBITMQ_USERNAME']

TRAINING_SERVICE_RABBITMQ_PASSWORD = default_config.TRAINING_SERVICE_RABBITMQ_PASSWORD
if os.environ['TRAINING_SERVICE_RABBITMQ_PASSWORD'] is not None:
    TRAINING_SERVICE_RABBITMQ_PASSWORD = os.environ['TRAINING_SERVICE_RABBITMQ_PASSWORD']

TRAINING_SERVICE_RABBITMQ_SERVER = default_config.TRAINING_SERVICE_RABBITMQ_SERVER
if os.environ['TRAINING_SERVICE_RABBITMQ_SERVER'] is not None:
    TRAINING_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_SERVICE_RABBITMQ_SERVER']

TRAINING_SERVICE_RABBITMQ_PORT = default_config.TRAINING_SERVICE_RABBITMQ_PORT
TRAINING_SERVICE_RABBITMQ_VHOST = default_config.TRAINING_SERVICE_RABBITMQ_VHOST
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
TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_EXCHANGE
TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAINING_REQUEST_ROUTING_KEY
TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_TRAIN_REQUEST_TASK_QUEUE
TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_VHOST
TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_RABBITMQ_URI

TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME
if os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME'] is not None:
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_USERNAME']

TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD
if os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD'] is not None:
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_PASSWORD']

TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER
if os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER'] is not None:
    TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER = os.environ['TRAINING_PROCESSOR_SERVICE_RABBITMQ_SERVER']

TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_PORT
TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST = default_config.TRAINING_PROCESSOR_SERVICE_RABBITMQ_VHOST
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
CLASSIFICATION_SERVER = default_config.CLASSIFICATION_SERVER
SERVER_PORT = default_config.SERVER_PORT
SERVER_URI = default_config.SERVER_URI


