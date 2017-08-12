import os
import fnmatch
import struct
import numpy
from PIL import Image
from array import *
import logging
# from datetime import datetime  # used when dumping raw transforms to disk
import dicebox_config as config
import filesystem_connecter
import requests
import json

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
        if SensoryInterface.InterfaceRole == 'client':
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

            # small batch approach
            response = self.make_sensory_api_call('api/sensory/request', json_data, 'POST')
            image_labels = response['labels']
            image_data = response['data']
            return image_data, image_labels

        elif SensoryInterface.InterfaceRole == 'server':
            return SensoryInterface.fsc.get_batch(batch_size, noise=noise)
        return None



    def make_sensory_api_call(self, end_point, json_data, call_type):
        headers = {
            'Content-type': 'application/json',
            'API-ACCESS-KEY': config.API_ACCESS_KEY,
            'API-VERSION': config.API_VERSION
        }
        try:
            url = "%s%s:%i/%s" % (config.SERVER_URI, config.SENSORY_SERVER, config.SERVER_PORT, end_point)
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