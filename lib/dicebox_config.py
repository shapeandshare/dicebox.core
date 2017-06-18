###############################################################################
# Configuration Options
###############################################################################
BATCH_SIZE = 10
NETWORK_NAME = 'dicebox_60x50'

DATA_BASE_DIRECTORY = 'datasets'
DATA_DIRECTORY = "%s/%s/data/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
CHECKPOINT_DIRECTORY = "%s/%s/checkpoint/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
CHECKPOINT_FILE = 'model.ckpt'
TENSORBOARD_LOG_DIRECTORY = "%s/%s/logs/" % (DATA_BASE_DIRECTORY, NETWORK_NAME)
TENSORBOARD_LOGGING = False

