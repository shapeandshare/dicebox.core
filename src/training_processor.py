"""
Training Processor
Copyright (c) 2017-2021 Joshua Burt
"""

import logging
import os
import errno
import json
import uuid
from datetime import datetime
from typing import Any

from tqdm import tqdm

from shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from shapeandshare.dicebox.factories.network_factory import NetworkFactory
from shapeandshare.dicebox.models.dicebox_network import DiceboxNetwork
from shapeandshare.dicebox.models.network import Network
from shapeandshare.dicebox.utils.helpers import make_sure_path_exists

# Config
config_file = "./projects/mnist/dicebox.training.config"
dicebox_config = DiceboxConfig(config_file=config_file)

"""
Setup logging
"""
make_sure_path_exists(dicebox_config.LOGS_DIR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
    filemode="w",
    # filename="%s/trainingprocessor.%s.log" % (dicebox_config.LOGS_DIR, os.environ["COMPUTERNAME"]),
    filename="%s/trainingprocessor.%s.log" % (dicebox_config.LOGS_DIR, os.uname()[1]),
)

def lonestar() -> object:
    genome = {
                "input_shape": [
                    28,
                    28,
                    3
                ],
                "output_size": 10,
                "optimizer": "adagrad",
                "layers": [
                    {
                        "type": "dense",
                        "size": 233,
                        "activation": "softplus"
                    },
                    {
                        "type": "dense",
                        "size": 108,
                        "activation": "tanh"
                    },
                    {
                        "type": "dense",
                        "size": 222,
                        "activation": "tanh"
                    },
                    {
                        "type": "flatten"
                    },
                    {
                        "type": "dropout",
                        "rate": 0.49411764705882355
                    },
                    {
                        "type": "dense",
                        "size": 174,
                        "activation": "tanh"
                    },
                    {
                        "type": "flatten"
                    }
                ]
            }
    return genome

# def lonestar() -> object:
#     return {
#         "input_shape": [28, 28, 3],
#         "output_size": 10,
#         "optimizer": "adagrad",
#         "layers": [
#             {"type": "flatten"},
#             {"type": "dense", "size": 128, "activation": "relu"},
#             {"type": "dropout", "rate": 0.01},
#         ],
#     }

# def lonestar() -> Any:
#     return {
#         "input_shape": [28, 28, 3],
#         "output_size": 10,
#         "optimizer": "adagrad",
#         "layers": [
#             {"type": "dropout", "rate": 0.6274509803921569},
#             {"type": "dense", "size": 533, "activation": "sigmoid"},
#             {"type": "dense", "size": 902, "activation": "elu"},
#         ],
#     }


    """
    Training Logic
    """


def main():
    training_request_id = str(uuid.uuid4())

    logging.debug("-" * 80)
    logging.debug("processing training request id: (%s)" % training_request_id)
    logging.debug("-" * 80)

    # create network factory
    network_factory: NetworkFactory = NetworkFactory(config=dicebox_config)
    network: Network = network_factory.create_network(network_definition=lonestar())
    dicebox_network: DiceboxNetwork = DiceboxNetwork(
        config=dicebox_config, optimizer=network.get_optimizer(), layers=network.get_layers()
    )
    del network

    if dicebox_config.LOAD_BEST_WEIGHTS_ON_START is True:
        logging.debug("-" * 80)
        logging.debug("attempting to restart training from previous session..")
        logging.debug("-" * 80)
        dicebox_network.load_model_weights(
            filename=f"{dicebox_config.WEIGHTS_DIR}/{dicebox_config.MODEL_WEIGHTS_FILENAME}"
        )
        logging.debug("-" * 80)
        logging.debug("Done")
        logging.debug("-" * 80)

    # network = DiceboxNetwork(CONFIG.NN_PARAM_CHOICES, True)
    # individual_network: Network = self.create_network(network_definition=individual_genome)
    # individual_dicebox_network = self.build_dicebox_network(network=individual_network)

    # if CONFIG.LOAD_BEST_WEIGHTS_ON_START is True:
    #     logging.debug('-' * 80)
    #     logging.debug('attempting to restart training from previous session..')
    #     logging.debug('-' * 80)
    #     network.create_lonestar(create_model=True,
    #                             weights_filename="%s/%s" % (CONFIG.WEIGHTS_DIR, CONFIG.MODEL_WEIGHTS_FILENAME))
    #     logging.debug('-' * 80)
    #     logging.debug('Done')
    #     logging.debug('-' * 80)
    # else:
    #     logging.debug('-' * 80)
    #     logging.debug('creating model, but NOT loading previous weights.')
    #     logging.debug('-' * 80)
    #     network.create_lonestar(create_model=True)
    #     logging.debug('-' * 80)
    #     logging.debug('Done')
    #     logging.debug('-' * 80)
    #

    # if dicebox_network.__fsc is not None:  # we only have fsc at this point, the sensory service is deprecated until more performant.

    logging.debug("-" * 80)
    logging.debug("writing category map to %s for later use with the weights.", dicebox_config.TMP_DIR)
    logging.debug("-" * 80)
    make_sure_path_exists(dicebox_config.WEIGHTS_DIR)
    with open("%s/category_map.json" % dicebox_config.WEIGHTS_DIR, "w") as category_mapping_file:
        category_mapping_file.write(json.dumps(dicebox_network.get_category_map()))

    i = 1
    pbar = tqdm(total=dicebox_config.EPOCHS)
    while i <= dicebox_config.EPOCHS:
        logging.debug("-" * 80)
        logging.debug("epoch (%i of %i)" % (i, dicebox_config.EPOCHS))
        logging.debug("-" * 80)
        # dicebox_network.train_and_save(dicebox_config.DATASET)
        # dicebox_network.compile()
        dicebox_network.train(update_accuracy=True)

        make_sure_path_exists(dicebox_config.WEIGHTS_DIR)
        logging.debug("-" * 80)
        logging.debug(f"dicebox_config.WEIGHTS_DIR={dicebox_config.WEIGHTS_DIR}")
        logging.debug(f"training_request_id={training_request_id}")
        logging.debug(f"accuracy={dicebox_network.get_accuracy() * 100}")
        full_path = "%s/%s.%.2f.%s.tf" % (
            dicebox_config.WEIGHTS_DIR,
            training_request_id,
            (dicebox_network.get_accuracy() * 100),
            datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        logging.debug("saving model weights after epoch %i to file %s" % (i, full_path))
        logging.debug("-" * 80)
        # dicebox_config.save_model(full_path)
        dicebox_network.save_model_weights(filename=full_path)

        # the next epoch..
        i += 1
        pbar.update(1)

    pbar.close()

    logging.debug("-" * 80)
    logging.debug("network accuracy: %.2f%%" % (dicebox_network.get_accuracy() * 100))
    logging.debug("-" * 80)

    # dicebox_network.print_network()
    # dicebox_network.decompile()
    logging.debug(dicebox_network.decompile())
    logging.debug("-" * 80)


if __name__ == "__main__":
    main()
