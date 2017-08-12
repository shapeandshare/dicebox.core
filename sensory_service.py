#!flask/bin/python
from lib import dicebox_config as config
from lib import sensory_interface
from flask import Flask, jsonify, request, make_response, abort
from flask_cors import CORS, cross_origin
import base64
import logging
import json
from datetime import datetime
import os
import errno
import uuid
import numpy

# import pika

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filemode='w',
    filename="%s/sensory_service.log" % config.LOGS_DIR
)

# Generate our Sensory Service Interface
ssc = sensory_interface.SensoryInterface('server')

# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def sensory_store(tmp_dir, data_dir, data_category, raw_image_data):
    filename = "%s" % datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.tmp.png')
    path = "%s/%s%s/" % (tmp_dir, data_dir, data_category)
    full_filename = "%s%s" % (path, filename)
    logging.debug("(%s)" % (full_filename))
    make_sure_path_exists(path)
    with open(full_filename, 'wb') as f:
        f.write(raw_image_data)
    return True

def sensory_request(batch_size, noise=0):
    #sensory_batch_request_id = uuid.uuid4()

    # TODO: dynamicly use the values coming in. :P
    data, labels = ssc.get_batch(batch_size, noise)

    #return sensory_batch_request_id
    return data, labels


app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:*"}})

@app.route('/api/sensory/store', methods=['POST'])
def make_api_sensory_store_public():
    if request.headers['API-ACCESS-KEY'] != config.API_ACCESS_KEY:
        abort(401)
    if request.headers['API-VERSION'] != config.API_VERSION:
        abort(400)
    if not request.json:
        abort(400)

    if 'name' not in request.json:
        abort(400)
    if 'width' not in request.json:
        abort(400)
    if 'height' not in request.json:
        abort(400)
    if 'category' not in request.json:
        abort(400)
    if 'data' not in request.json:
        abort(400)
    if 'data' in request.json and type(request.json['data']) != unicode:
        abort(400)

    dataset_name = request.json.get('name')
    image_width = request.json.get('width')
    image_height = request.json.get('height')
    category = request.json.get('category')
    encoded_image_data = request.json.get('data')
    decoded_image_data = base64.b64decode(encoded_image_data)

    network_name = "%s_%ix%i" % (dataset_name, image_width, image_height)
    data_directory = "%s/%s/data/" % (config.DATA_BASE_DIRECTORY, network_name)

    return_code = sensory_store(config.TMP_DIR, data_directory, category, decoded_image_data)
    return make_response(jsonify({'sensory_store': return_code}), 201)

# for small batches..
@app.route('/api/sensory/request', methods=['POST'])
def make_api_sensory_request():
    if request.headers['API-ACCESS-KEY'] != config.API_ACCESS_KEY:
        logging.debug('bad access key')
        abort(401)
    if request.headers['API-VERSION'] != config.API_VERSION:
        logging.debug('bad access version')
        abort(400)
    if not request.json:
        logging.debug('request not json')
        abort(400)

    if 'batch_size' not in request.json:
        logging.debug('batch size not in request')
        abort(400)
    if 'noise' not in request.json:
        logging.debug('noise not in request')
        abort(400)

    batch_size = request.json.get('batch_size')
    noise = request.json.get('noise')

    # sensory_batch_request_id = sensory_request(dataset_name, image_width, image_height, batch_size)
    data, labels = sensory_request(batch_size, noise)
    return make_response(jsonify({
                                  'labels': numpy.array(labels).tolist(),
        'data': numpy.array(data).tolist()
                                  }), 201)


@app.route('/api/version', methods=['GET'])
def make_api_version_public():
    return make_response(jsonify({'version':  str(config.API_VERSION)}), 201)


@app.route('/health/plain', methods=['GET'])
@cross_origin()
def make_health_plain_public():
    return make_response('true', 201)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    logging.debug('starting flask app')
    app.run(debug=config.FLASK_DEBUG, host=config.LISTENING_HOST, threaded=True)
