import base64
from typing import Any

import requests
import json  # for writing category data to file
import logging
import os
import errno

from shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from shapeandshare.dicebox.connectors.filesystem_connector import FileSystemConnector
from shapeandshare.dicebox.factories.network_factory import NetworkFactory
from shapeandshare.dicebox.models.dicebox_network import DiceboxNetwork
from shapeandshare.dicebox.models.network import Network
from shapeandshare.dicebox.utils.helpers import make_sure_path_exists

config_file = "./projects/mnist/dicebox.config"
dicebox_config = DiceboxConfig(config_file=config_file)

###############################################################################
# Setup logging.
###############################################################################
make_sure_path_exists(dicebox_config.LOGS_DIR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
    filemode="w",
    filename="%s/mnist_test_client.%s.log" % (dicebox_config.LOGS_DIR, os.uname()[1]),
)


def build_dicebox_network(config: DiceboxConfig, network: Network) -> DiceboxNetwork:
    return DiceboxNetwork(
        config=config, optimizer=network.get_optimizer(), layers=network.get_layers(), disable_data_indexing=True
    )


def lonestar() -> object:
    return {
        "input_shape": [28, 28, 3],
        "output_size": 10,
        "optimizer": "adagrad",
        "layers": [
            {"type": "flatten"},
            {"type": "dense", "size": 128, "activation": "relu"},
            {"type": "dropout", "rate": 0.01},
        ],
    }

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


def get_category_map():
    jdata = {}
    if len(jdata) == 0:
        with open(f"{dicebox_config.WEIGHTS_DIR}/category_map.json") as data_file:
            raw_cat_data = json.load(data_file)
        for d in raw_cat_data:
            jdata[str(raw_cat_data[d])] = str(d)
        print("loaded category map from file.")

    # print(jdata)
    return jdata


###############################################################################
# Method call that creates the classification response.
# We are loading a specific model (lonestar), with a pre-created weights file.
###############################################################################
def get_classification(image_data) -> int:
    classification = dicebox_network.classify(network_input=image_data).tolist()
    logging.info("classification: (%s)", classification)
    return classification[0]


# def make_api_call(end_point, json_data, call_type):
#     headers = {
#         "Content-Type": "application/json",
#         "API-ACCESS-KEY": dicebox_config.API_ACCESS_KEY,
#         "API-VERSION": dicebox_config.API_VERSION,
#     }
#     try:
#         url = "%s%s:%i/%s" % (
#             dicebox_config.CLASSIFICATION_SERVER_URI,
#             dicebox_config.CLASSIFICATION_SERVER,
#             dicebox_config.CLASSIFICATION_SERVER_PORT,
#             end_point,
#         )
#         # print(f"calling {url}")
#         response = None
#         if call_type == "GET":
#             response = requests.get(url, data=json_data, headers=headers)
#         elif call_type == "POST":
#             # print(json_data)
#             response = requests.post(url, data=json_data, headers=headers)
#
#         if response is not None:
#             if response.status_code != 500:
#                 return response.json()
#     except Exception as error:
#         print(f"Error: {str(error)}")
#         return {}
#     return {}


###############################################################################
# prep our data sets

###############################################################################
# Create the network. (Create FSC, disabling data indexing)
###############################################################################
network_factory: NetworkFactory = NetworkFactory(config=dicebox_config)
network: Network = network_factory.create_network(network_definition=lonestar())
dicebox_network: DiceboxNetwork = build_dicebox_network(config=dicebox_config, network=network)
print("|| Compiling ..")
dicebox_network.compile()
print("|| Loading model..")
dicebox_network.load_model_weights(f"{dicebox_config.WEIGHTS_DIR}/{dicebox_config.MODEL_WEIGHTS_FILENAME}")


###############################################################################
# Load categories for the model
###############################################################################
with open("%s/category_map.json" % dicebox_config.WEIGHTS_DIR) as data_file:
    jdata = json.load(data_file)
server_category_map = {}
for d in jdata:
    server_category_map[str(jdata[d])] = str(d)
logging.info("loaded categories: %s" % server_category_map)


print("Creating FileSystem Data Connector")
fsc = FileSystemConnector(
    config=dicebox_config, data_directory=dicebox_config.DATA_DIRECTORY, disable_data_indexing=False
)
print("Loading Data Set")
network_input_index = fsc.get_data_set()
# print("Network Input Index: %s" % network_input_index)

# Get our classification categories
server_category_map = get_category_map()
print("Category Map: %s" % server_category_map)

##############################################################################
# Evaluate the Model
##############################################################################

summary_fail = 0
summary_success = 0

count = 0
for index in network_input_index:
    metadata = network_input_index[index]

    filename = "%s%s" % (dicebox_config.DATA_DIRECTORY, index)
    print(f"filename: {filename}")
    with open(filename, "rb") as file:
        file_content: bytes = file.read()

    base64_encoded_content: bytes = base64.b64encode(file_content)
    # base64_encoded_string: str = base64_encoded_content.decode("utf-8")

    prediction = get_classification(file_content)
    category = server_category_map[str(prediction)]

    if category == metadata[1]:
        print("correct!")
        summary_success += 1
    else:
        print(f"FAIL - Expected {metadata[1]}, but received {category}")
        summary_fail += 1

    if count >= 10000:
        count += 1
        break
    else:
        count += 1


success_percentage = (float(summary_success) / count) * 100
failure_percentage = (float(summary_fail) / count) * 100

print("summary")
print("success: (%i)" % summary_success)
print("failures: (%i)" % summary_fail)
print("total tests: (%i)" % count)
print("success rate: (%f)" % success_percentage)
print("failure rate: (%f)" % failure_percentage)
