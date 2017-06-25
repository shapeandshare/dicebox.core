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

NB_CLASSES = 5
BATCH_SIZE = 6000
INPUT_SHAPE = (3000,)
NOISE = 0.9
TRAIN_BATCH_SIZE = 6000
TEST_BATCH_SIZE = 2000
