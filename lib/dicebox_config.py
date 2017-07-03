###############################################################################
# Configuration Options
###############################################################################

NETWORK_NAME = 'dicebox_60x50'
DATASET = 'dicebox'
NB_CLASSES = 11
INPUT_SHAPE = (3000,)

DATA_BASE_DIRECTORY = 'datasets'
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)

NN_PARAM_CHOICES = {
    'nb_neurons': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
    'nb_layers': [1, 2, 3, 5, 8, 13, 21],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam'],
}

EPOCHS = 10000
GENERATIONS = 100  # Number of times to evole the population.
POPULATION = 50  # Number of networks in each generation.
BATCH_SIZE = 5000
NOISE = 0.3
TRAIN_BATCH_SIZE = 120000
TEST_BATCH_SIZE = 10000

LOGS_DIR='./logs'
WEIGHTS_DIR='./weights'

###############################################################################
# Server Configuration Options
###############################################################################
API_ACCESS_KEY = '6{t}*At&R;kbgl>Mr"K]=F+`EEe'
API_VERSION = '0.1.0'
LISTENING_HOST='0.0.0.0'
FLASK_DEBUG=False
MODEL_WEIGHTS_FILENAME='weights.best.hdf5'
