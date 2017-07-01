import logging
import lib.dicebox_config as config
from lib.network import Network
from datetime import datetime

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filemode='w',
    filename='lonestar_train.log'
)

def main():
    nn_param_choices = config.NN_PARAM_CHOICES
    dataset = config.DATASET

    network = Network(nn_param_choices)
    network.create_lonestar()
    i = 1
    while i <= config.EPOCHS:
        logging.info("epoch (%i of %i)" % (i, config.EPOCHS))
        network.train_and_save(dataset)

        # save the model after every epoch, regardless of accuracy.
        filename = "weights.epoch_%i.final.%s.hdf5" % (i, datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f'))
        logging.info("saving model weights after epoch %i to file %s" % (i, filename))
        network.save_model(filename)

        # the next epoch..
        i += 1

    logging.info("network accuracy: %.2f%%" % (network.accuracy * 100))
    logging.info('-'*80)
    network.print_network()

if __name__ == '__main__':
    main()
