import ConfigParser
import json

my_config = ConfigParser.ConfigParser()
configFilePath = r'./dicebox.config'
my_config.read(configFilePath)


###############################################################################
# Data Set Options
###############################################################################

# Load user defined config
DATASET = my_config.get('DATASET', 'name')
NB_CLASSES = my_config.getint('DATASET', 'categories')
IMAGE_WIDTH = my_config.getint('DATASET', 'image_width')
IMAGE_HEIGHT = my_config.getint('DATASET', 'image_height')
DATA_BASE_DIRECTORY = my_config.get('DIRECTORY', 'dataset_base_directory')

# Build Calculated Configs
NETWORK_NAME = "%s_%ix%i" % (DATASET, IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH*IMAGE_HEIGHT,)
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)


###############################################################################
# Neural Network Taxonomy Options
###############################################################################
NN_PARAM_CHOICES = {
    'nb_neurons': json.loads(my_config.get('TAXONOMY', 'neurons')),
    'nb_layers': json.loads(my_config.get('TAXONOMY', 'layers')),
    'activation': json.loads(my_config.get('TAXONOMY', 'activation')),
    'optimizer': json.loads(my_config.get('TAXONOMY', 'optimizer'))
}


###############################################################################
# Lonestar Options
# {'nb_layers': 2, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 987}
# 07/03/2017 04:02:49 AM - INFO - Network accuracy: 97.27%
###############################################################################
NN_LONESTAR_PARAMS = {
    'nb_neurons': my_config.getint('LONESTAR', 'neurons'),
    'nb_layers': my_config.getint('LONESTAR', 'layers'),
    'activation': my_config.get('LONESTAR', 'activation'),
    'optimizer': my_config.get('LONESTAR', 'optimizer')
}


###############################################################################
# Evolution Options
###############################################################################
EPOCHS = my_config.getint('EVOLUTION', 'epochs')
GENERATIONS = my_config.getint('EVOLUTION', 'generations')  # Number of times to evole the population.
POPULATION = my_config.getint('EVOLUTION', 'population')  # Number of networks in each generation.
NOISE = my_config.getfloat('GLOBAL', 'noise')


###############################################################################
# Training Options / Settings for the 1920x1080 dataset
###############################################################################
BATCH_SIZE = my_config.getint('TRAINING', 'batch_size')
TRAIN_BATCH_SIZE = my_config.getint('TRAINING', 'train_batch_size')
TEST_BATCH_SIZE = my_config.getint('TRAINING', 'test_batch_size')


###############################################################################
# Direcrtory Options
###############################################################################
LOGS_DIR = my_config.get('DIRECTORY', 'logs_dir')
WEIGHTS_DIR = my_config.get('DIRECTORY', 'weights_dir')
TMP_DIR = my_config.get('DIRECTORY', 'tmp_dir')


###############################################################################
# Server Options
###############################################################################
API_ACCESS_KEY = my_config.get('SERVER', 'api_access_key')
API_VERSION = my_config.get('SERVER', 'api_version')
LISTENING_HOST = my_config.get('SERVER', 'listening_host')
FLASK_DEBUG = my_config.getboolean('SERVER', 'flask_debug')
MODEL_WEIGHTS_FILENAME = my_config.get('SERVER', 'model_weights_filename')

###############################################################################
# Client Options
###############################################################################
CLASSIFICATION_SERVER = my_config.get('CLIENT', 'classification_server')
SERVER_PORT = my_config.getint('CLIENT', 'classification_port')
SERVER_URI = my_config.get('CLIENT', 'uri')
