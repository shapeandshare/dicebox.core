###############################################################################
# Configuration Options
###############################################################################

DATASET = 'dicebox'
NB_CLASSES = 11
IMAGE_WIDTH=480
IMAGE_HEIGHT=270
NETWORK_NAME = "%s_%ix%i" % (DATASET, IMAGE_WIDTH, IMAGE_HEIGHT)
INPUT_SHAPE = (IMAGE_WIDTH*IMAGE_HEIGHT,)

DATA_BASE_DIRECTORY = 'datasets'
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)

NN_PARAM_CHOICES = {
    'nb_neurons': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
    'nb_layers': [1, 2, 3, 5, 8, 13, 21],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam']
}

# {'nb_layers': 2, 'activation': 'sigmoid', 'optimizer': 'adamax', 'nb_neurons': 987}
#07/03/2017 04:02:49 AM - INFO - Network accuracy: 97.27%
NN_LONESTAR_PARAMS = {
    'nb_neurons': 987,
    'nb_layers': 2,
    'activation': 'sigmoid',
    'optimizer': 'adamax'
}

EPOCHS = 10000
GENERATIONS = 100  # Number of times to evole the population.
POPULATION = 50  # Number of networks in each generation.
NOISE = 0.1

# Settings for the 1920x1080 dataset
BATCH_SIZE = 150
TRAIN_BATCH_SIZE = 1000
TEST_BATCH_SIZE = 200

# Settings for the 60x50 dataset
#BATCH_SIZE = 15000
#TRAIN_BATCH_SIZE = 212000
#TEST_BATCH_SIZE = 15000

LOGS_DIR='./logs'
WEIGHTS_DIR='./weights'
TMP_DIR='./tmp'

###############################################################################
# Server Configuration Options
###############################################################################
API_ACCESS_KEY = '6{t}*At&R;kbgl>Mr"K]=F+`EEe'
API_VERSION = '0.1.0'
LISTENING_HOST='0.0.0.0'
FLASK_DEBUG=False
MODEL_WEIGHTS_FILENAME='weights.best.hdf5'
