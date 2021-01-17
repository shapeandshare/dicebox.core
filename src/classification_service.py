#!flask/bin/python
###############################################################################
# Classification Service
#   Handles requests for classification of data from a client.
#
# Copyright (c) 2017-2019 Joshua Burt
###############################################################################


###############################################################################
# Dependencies
###############################################################################
from flask import Flask, jsonify, request, make_response, abort
from flask_cors import CORS, cross_origin
import base64
import logging
import json
import os
import errno

# from dicebox.config.dicebox_config import DiceboxConfig
# from dicebox.dicebox_network import DiceboxNetwork

# Config
# from flask import Flask

from shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from shapeandshare.dicebox.factories.network_factory import NetworkFactory
from shapeandshare.dicebox.models.dicebox_network import DiceboxNetwork
from shapeandshare.dicebox.models.network import Network
from shapeandshare.dicebox.utils.helpers import make_sure_path_exists

config_file = "./projects/mnist/dicebox.config"
dicebox_config = DiceboxConfig(config_file)


def build_dicebox_network(config: DiceboxConfig, network: Network) -> DiceboxNetwork:
    return DiceboxNetwork(config=config, optimizer=network.get_optimizer(), layers=network.get_layers())


def lonestar() -> Any:
    return {
        "input_shape": [28, 28, 3],
        "output_size": 10,
        "optimizer": "sgd",
        "layers": [{"type": "dense", "size": 427, "activation": "softmax"}],
    }


###############################################################################
# Setup logging.
###############################################################################
make_sure_path_exists(dicebox_config.LOGS_DIR)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
    filemode="w",
    filename="%s/classificationservice.%s.log" % (dicebox_config.LOGS_DIR, os.environ["COMPUTERNAME"]),
)

###############################################################################
# Create the network. (Create FSC, disabling data indexing)
###############################################################################
network_factory: NetworkFactory = NetworkFactory(config=dicebox_config)
network: Network = network_factory.create_network(network_definition=lonestar())
dicebox_network: DiceboxNetwork = build_dicebox_network(config=dicebox_config, network=network)
dicebox_network.load_model_weights()

###############################################################################
# Load categories for the model
###############################################################################
with open("%s/category_map.json" % dicebox_config.WEIGHTS_DIR) as data_file:
    jdata = json.load(data_file)
server_category_map = {}
for d in jdata:
    server_category_map[str(jdata[d])] = str(d)
logging.info("loaded categories: %s" % server_category_map)

###############################################################################
# Create the flask, and cors config
###############################################################################
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:*"}})


###############################################################################
# Method call that creates the classification response.
# We are loading a specific model (lonestar), with a pre-created weights file.
# Here we elect to create_model, which will also load the model weights on
# first use.
###############################################################################
def get_classification(image_data):
    try:
        network.create_lonestar(
            create_model=True,
            weights_filename="%s/%s" % (dicebox_config.WEIGHTS_DIR, dicebox_config.MODEL_WEIGHTS_FILENAME)
        )
    except:
        logging.error("Error summoning lonestar")
        return -1

    classification = network.classify(image_data)
    logging.info("classification: (%s)", classification)
    return classification[0]


# We need more specificity before we start catching stuff here, otherwise this
# catches too many issues.
#
#    try:
#        classification = network.classify(image_data)
#        logging.info("classification: (%s)" % classification)
#        return classification[0]
#    except:
#        logging.error('Error making prediction..')
#        return -1


###############################################################################
# Provides categories for client consumption
###############################################################################
@app.route("/api/category", methods=["GET"])
def make_api_categorymap_public():
    if request.headers["API-ACCESS-KEY"] != dicebox_config.API_ACCESS_KEY:
        abort(403)
    if request.headers["API-VERSION"] != dicebox_config.API_VERSION:
        abort(400)

    return make_response(jsonify({"category_map": server_category_map}), 200)


###############################################################################
# Endpoint which provides classification of image data.
###############################################################################
@app.route("/api/classify", methods=["POST"])
def make_api_get_classify_public():
    if request.headers["API-ACCESS-KEY"] != dicebox_config.API_ACCESS_KEY:
        abort(403)
    if request.headers["API-VERSION"] != dicebox_config.API_VERSION:
        abort(400)
    if not request.json:
        abort(400)
    if "data" in request.json and type(request.json["data"]) != unicode:
        abort(400)

    encoded_image_data = request.json.get("data")
    decoded_image_data = base64.b64decode(encoded_image_data)
    classification = get_classification(decoded_image_data)

    return make_response(jsonify({"classification": classification}), 200)


###############################################################################
# Returns API version
###############################################################################
@app.route("/api/version", methods=["GET"])
def make_api_version_public():
    return make_response(jsonify({"version": str(dicebox_config.API_VERSION)}), 200)


###############################################################################
# Super generic health end-point
###############################################################################
@app.route("/health/plain", methods=["GET"])
@cross_origin()
def make_health_plain_public():
    return make_response("true", 200)


###############################################################################
# 404 Handler
###############################################################################
@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}), 404)


###############################################################################
# main entry point, thread safe
###############################################################################
if __name__ == "__main__":
    logging.debug("starting flask app")
    app.run(debug=dicebox_config.FLASK_DEBUG, host=dicebox_config.LISTENING_HOST, threaded=False)
