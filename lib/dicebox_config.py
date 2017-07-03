###############################################################################
# Configuration Options
###############################################################################
#BATCH_SIZE = 1
NETWORK_NAME = 'dicebox_60x50'

DATA_BASE_DIRECTORY = 'datasets'
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
#CHECKPOINT_DIRECTORY = "%s/%s/checkpoint/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
#CHECKPOINT_FILE = 'model.ckpt'
#TENSORBOARD_LOG_DIRECTORY = "%s/%s/logs/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
#TENSORBOARD_LOGGING = False


EPOCHS = 10000
GENERATIONS = 100  # Number of times to evole the population.
POPULATION = 50  # Number of networks in each generation.
DATASET = 'dicebox'

NN_PARAM_CHOICES = {
    'nb_neurons': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
    'nb_layers': [1, 2, 3, 5, 8, 13, 21],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam'],
}

NB_CLASSES = 11
BATCH_SIZE = 5000
INPUT_SHAPE = (3000,)
NOISE = 0.3
TRAIN_BATCH_SIZE = 120000
TEST_BATCH_SIZE = 10000
