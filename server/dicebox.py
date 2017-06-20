#!flask/bin/python
from lib import dicebox_config as config
from flask import Flask, jsonify, request, make_response, abort
import sys
import base64
import os
from datetime import datetime

from PIL import Image
from array import *
import numpy
import logging

from lib.optimizer import Optimizer

VERSION = '0.1.0'

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filemode='w',
    filename='dicebox_server.log'
)

# Allow us to load it up in tensorboard
# file_writer = tf.summary.FileWriter('./logs', sess.graph)

population = 1  # Number of networks in each generation.
dataset = 'dicebox'

nn_param_choices = {
    'nb_neurons': [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597],
    'nb_layers': [1, 2, 3, 5, 8, 13, 21],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                  'adadelta', 'adamax', 'nadam'],
}

optimizer = Optimizer(nn_param_choices)
networks = optimizer.create_lonestar(population)


def process_image(image_data):
    # ugh dump to file for the time being
    filename = "./tmp/%s" % datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png')
    with open(filename, 'wb') as f:
        f.write(image_data)

    pixel_data = array('B')

    #print('converting payload into an image object')
    #m2 = Image.frombytes('RGB', IMAGE_SIZE, io.BytesIO.load(image_data), decoder_name='raw')

    Im = Image.open(filename)
    pixel = Im.load()
    os.remove(filename)

    width, height = Im.size

    for x in range(0, width):
        for y in range(0, height):
            pixel_data.append(pixel[x, y])

    data = numpy.frombuffer(pixel_data, dtype=numpy.uint8)
    return data


def train_networks(networks, dataset):
    """Train each network.

    Args:
        networks (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    """
    #pbar = tqdm(total=len(networks))
    for network in networks:
        # network.load_n_score(dataset)
        #network.load_n_score_single(dataset)
        return network.load_n_predict_single(dataset)
        #pbar.update(1)
    #pbar.close()

def get_prediction(image_data):
    try:
        #network_input = process_image(image_data)
        #prepared_input = [ network_input ]
        #network_output = sess.run(cnn.network_output, feed_dict={cnn.x: prepared_input, cnn.keep_prob: 1.0})
        #output = network_output[0].tolist()
        #return output
        logging.info('get_prediction placeholder called..')

        # Train and get accuracy for networks.
        #dataset = numpy.array(image_data)
        #dataset /= 255
        dataset = image_data
        logging.info(dataset)
        prediction = train_networks(networks, dataset)
        logging.info("prediction class: (%s)" % prediction)
        return prediction

    except:
        print("Error making prediction.")
        raise
        return {}


app = Flask(__name__)

@app.route('/api/prediction', methods=['POST'])
def make_api_prediction_public():
    if request.headers['API-ACCESS-KEY'] != '6{t}*At&R;kbgl>Mr"K]=F+`EEe':
        abort(401)
    if request.headers['API-VERSION'] != VERSION:
        abort(400)
    if not request.json:
        abort(400)
    if 'data' in request.json and type(request.json['data']) != unicode:
        abort(400)

    predication_data = request.json.get('data')
    decoded_image_data = base64.b64decode(predication_data)
    prediction = get_prediction(decoded_image_data)

    return make_response(jsonify({'prediction': prediction }), 201)


@app.route('/version', methods=['GET'])
def make_version_public():
    return make_response(jsonify({ 'version':  str(VERSION)}), 201)


@app.route('/health/plain', methods=['GET'])
def make_plain_health_public():
    return make_response('true', 201)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    print('starting flask app')
    app.run(debug=True,host='0.0.0.0', threaded=True)