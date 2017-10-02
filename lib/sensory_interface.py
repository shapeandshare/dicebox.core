import os
import fnmatch
import struct
import numpy
from PIL import Image
from array import *
import logging
# from datetime import datetime  # used when dumping raw transforms to disk
import docker_config as config
import filesystem_connecter
import requests
import json
import time
import base64
from datetime import datetime
import pika
import errno
import array
import itertools


class SensoryInterface:

    fsc = None  # file system connector
    InterfaceRole = None

    def __init__(self, role):
        logging.debug('Three wise monkeys');
        SensoryInterface.InterfaceRole = role

        # Consuming side
        if role == 'client':
            logging.debug("[%s] client side sensory interface code goes here." % SensoryInterface.InterfaceRole)

        if role == 'server':
            if SensoryInterface.fsc is None:
                logging.debug("[%s] creating a new fsc.." % SensoryInterface.InterfaceRole)
                SensoryInterface.fsc = filesystem_connecter.FileSystemConnector(config.DATA_DIRECTORY)


    def get_batch(self, batch_size=0, noise=0):
        logging.debug('-' * 80)
        logging.debug("get_batch(batch_size=%i, noise=%i)" % (batch_size, noise))
        logging.debug('-' * 80)

        if SensoryInterface.InterfaceRole == 'client':
            logging.debug('-' * 80)
            logging.debug('we are client')
            logging.debug('-' * 80)
            # TODO: We assemble our data through successive calls to the message service
            # or rest for small..

            outjson = {}
            outjson['batch_size'] = batch_size
            outjson['noise'] = noise

            json_data = json.dumps(outjson)
            batch_request_id = self.make_sensory_api_call('api/sensory/batch', json_data, 'POST')
            batch_id = batch_request_id['batch_id']

            logging.debug(batch_id)

            # the queue will stop existing if we take too long, or we clear all the messages.
            # call the sensory service to poll and pop our messages off, it will interface with rabbitmq for us

            outjson = {}
            outjson['batch_id'] = batch_id
            json_data = json.dumps(outjson)

            image_label = []
            image_data = []


            # natural_category_list = self.get_category_map()

            response = {}
            count = 0
            # while count != batch_size:
            #     logging.debug("count: %s" % count)
            #     response = self.make_sensory_api_call('api/sensory/poll', json_data, 'POST')
            #
            #     while (len(response) < 1):
            #         time.sleep(1)
            #         response = self.make_sensory_api_call('api/sensory/poll', json_data, 'POST')
            #
            #     # logging.debug(response)
            #     # batch_item = json.load(response)
            #
            #     # lets attempt to cache to file here and convert the one-hot value to the directory structure
            #     # first convert label to one-hot value
            #     cat_index = -1
            #     one_hot_cat = response['label']
            #     for i in range(0, len(one_hot_cat)):
            #         if one_hot_cat[i] == 1:
            #             cat_index = i
            #     if cat_index < 0:
            #         logging.debug('unable to decode one hot category value')
            #         raise
            #     else:
            #         logging.debug("decoded one hot category to: (%i)" % cat_index)
            #
            #     # look up human-readable category
            #     logging.debug("cat map: (%s)" % natural_category_list)
            #     current_category = natural_category_list[str(cat_index)]
            #     logging.debug("decoded natural category: (%s)" % current_category)
            #
            #     encoded_image_data = response.json.get('data')
            #     decoded_image_data = base64.b64decode(encoded_image_data)
            #     self.sensory_store('./tmp', current_category, decoded_image_data)
            #
            #     image_label.append(response['label']) #  = numpy.append(image_label,[response['label']])
            #     image_data.append(response['data'])  # = numpy.append(image_data, [response['data']])
            #     # image_label = [response['label']]
            #     # image_data = [response['data']]
            #
            #     #logging.debug(image_label)
            #     #logging.debug(image_data)
            #
            #     count += 1

            while count < batch_size:
                logging.debug("count: %s" % count)
                new_image_data = None
                new_image_label = None

                try:
                    new_image_data, new_image_label = self.sensory_batch_poll(batch_id)
                    image_data.append(new_image_data)
                    image_label.append(new_image_label)
                    count += 1
                except:
                    logging.debug('.')

                cat_index = -1
                if new_image_label is not None:
                    one_hot_cat = new_image_label
                    for i in range(0, len(one_hot_cat)):
                        if one_hot_cat[i] == 1:
                            cat_index = i

                # lets attempt to cache to file here and convert the one-hot value to the directory structure
                # first convert label to one-hot value
                if cat_index < 0:
                    logging.debug('unable to decode one hot category value')
                else:
                    logging.debug("decoded one hot category to: (%i)" % cat_index)

                    #decoded_image_data = base64.b64decode(new_image_data)
                    logging.debug(new_image_data)
                    decoded_image_data = base64.b64decode(''.join(str(x) for x in new_image_data))
                    logging.debug('raw image decoded, dumping to file ..')
                    ret = self.sensory_store(config.TMP_DIR, cat_index, decoded_image_data)
                    if ret is True:
                        logging.debug('successfully stored to disk..')
                    else:
                        logging.debug('failed to store to disk!')

            logging.debug('-' * 80)
            logging.debug('Done receiving batch.')
            logging.debug('-' * 80)
            return image_data, image_label

            # small batch approach
            #response = self.make_sensory_api_call('api/sensory/request', json_data, 'POST')
            #image_labels = response['labels']
            #image_data = response['data']
            #return image_data, image_labels

        elif SensoryInterface.InterfaceRole == 'server':
            logging.debug('-' * 80)
            logging.debug('we are server')
            logging.debug('-' * 80)
            return SensoryInterface.fsc.get_batch(batch_size, noise=noise)
        return None



    def make_sensory_api_call(self, end_point, json_data, call_type):
        headers = {
            'Content-type': 'application/json',
            'API-ACCESS-KEY': config.API_ACCESS_KEY,
            'API-VERSION': config.API_VERSION
        }

        try:
            url = "%s%s:%s/%s" % (config.SENSORY_URI, config.SENSORY_SERVER, config.SENSORY_PORT, end_point)
            # logging.debug('-' * 80)
            # logging.debug(url)
            # logging.debug('-' * 80)
            response = None
            if call_type == 'GET':
                response = requests.get(url, data=json_data, headers=headers)
            elif call_type == 'POST':
                response = requests.post(url, data=json_data, headers=headers)

            if response is not None:
                if response.status_code != 500:
                    return response.json()
        except:
            return {}
        return {}


    # TODO: temporary - we should calculate this using one of the provided methods this is really for testing the threaded-caching
    def get_category_map(self):
        jdata = {}
        with open('./category_map.json') as data_file:
            raw_cat_data = json.load(data_file)
        for d in raw_cat_data:
            jdata[str(raw_cat_data[d])] = str(d)
        return jdata

    # TODO: temporary - we should calculate this using one of the provided methods this is really for testing the threaded-caching
    def sensory_store(self, data_dir, data_category, raw_image_data):
        filename = "%s" % datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.png')
        path = "%s%s/" % (data_dir, data_category)
        full_filename = "%s%s" % (path, filename)
        logging.debug("(%s)" % (full_filename))
        self.make_sure_path_exists(path)
        with open(full_filename, 'wb') as f:
            f.write(raw_image_data)
        return True

    # TODO: temporary - we should calculate this using one of the provided methods this is really for testing the threaded-caching
    # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    def make_sure_path_exists(self, path):
        try:
            if os.path.exists(path) is False:
                os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


    def sensory_batch_poll(self, batch_id):
        # lets try to grab more than one at a time // combine and return
        # since this is going to clients lets reduce chatter

        data = None
        label = None

        url = config.SENSORY_SERVICE_RABBITMQ_URL
        # logging.debug(url)
        parameters = pika.URLParameters(url)

        connection = pika.BlockingConnection(parameters=parameters)

        channel = connection.channel()

        method_frame, header_frame, body = channel.basic_get(batch_id)
        if method_frame:
            #logging.debug("%s %s %s" % (method_frame, header_frame, body))
            message = json.loads(body)
            label = message['label']
            data = message['data']
            #logging.debug(label)
            #logging.debug(data)
            channel.basic_ack(method_frame.delivery_tag)
        else:
            logging.debug('no message returned')
        connection.close()
        return data, label

