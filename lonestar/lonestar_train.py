import logging
import lib.dicebox_config as config
from lib.network import Network

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filemode='w',
    filename='lonestar_train.log'
)

def main():
    nn_param_choices = config.NN_PARAM_CHOICES
    dataset = config.DATASET

    network = Network(nn_param_choices)
    network.create_lonestar()
    network.train_and_save(dataset)

    logging.info("Generation average: %.2f%%" % (network.accuracy * 100))
    logging.info('-'*80)
    network.print_network()

if __name__ == '__main__':
    main()
