import json
import logging
from datetime import datetime

import pika
import requests
from PIL import Image

from .filesystem_connector import FileSystemConnector
from ..config.dicebox_config import DiceboxConfig
from ..utils.helpers import make_sure_path_exists


class SensoryServiceConnector:
    fsc = None  # file system connector
    interface_role = None
    config = None

    def __init__(self, role, config: DiceboxConfig):
        logging.debug('Three wise monkeys')

        self.config = config

        if self.interface_role is None:
            self.interface_role = role

        # Consuming side
        if role == 'client':
            logging.debug("[%s] client side sensory interface code goes here.", self.interface_role)

        if role == 'server':
            if self.fsc is None:
                logging.debug("[%s] creating a new __fsc..", self.interface_role)
                self.fsc = FileSystemConnector(data_directory=self.config.DATA_DIRECTORY,
                                               config=self.config)

    def get_batch(self, batch_size=0, noise=0):
        logging.debug('-' * 80)
        logging.debug("get_batch(batch_size=%i, noise=%i)" % (batch_size, noise))
        logging.debug('-' * 80)

        if self.interface_role == 'client':
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

            # Check the value of the response!
            if batch_request_id is None or batch_request_id == {}:
                # Then we failed to contact the sensory service, or some other error occurred..
                logging.error('Error getting a sensory batch request id!')
                raise Exception('Error getting a sensory batch request id from the sensory service!')

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

                    cat_index = -1
                    if new_image_label is not None:
                        one_hot_cat = new_image_label
                        for i in range(0, len(one_hot_cat)):
                            if one_hot_cat[i] == 1:
                                cat_index = i

                    # lets attempt to cache to file here and convert the one-hot value to the directory structure
                    # first convert label to one-hot value
                    if cat_index < 0:
                        logging.error('unable to decode one hot category value')
                        raise Exception('unable to decode one hot category value')
                    else:
                        logging.debug("decoded one hot category to: (%i)" % cat_index)

                        newimage = Image.new('L', (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT))  # type, size
                        newimage.putdata(new_image_data)

                        #############################################################
                        # TODO: this enables caching the images to disk, for later use
                        #############################################################
                        # image review etc. Ideally we would also attempt to PULL from cache too
                        # however this has not yet been implemented...
                        #
                        # # logging.debug('raw image decoded, dumping to file ..')
                        # ret = self.image_sensory_store(self.CONFIG.TMP_DIR, cat_index, newimage)
                        # if ret is True:
                        #     logging.debug('successfully stored to disk..')
                        # else:
                        #     logging.error('failed to store to disk!')
                        #     raise Exception('failed to store to disk!')
                        #############################################################

                        image_data.append(new_image_data)
                        image_label.append(new_image_label)
                        count += 1
                except Exception as e:
                    logging.error(e)
                    logging.debug('.')
                    # raise e

            logging.debug('-' * 80)
            logging.debug('Done receiving batch.')
            logging.debug('-' * 80)
            return image_data, image_label

            # small batch approach
            # response = self.make_sensory_api_call('api/sensory/request', json_data, 'POST')
            # image_labels = response['labels']
            # image_data = response['data']
            # return image_data, image_labels

        elif self.interface_role == 'server':
            logging.debug('-' * 80)
            logging.debug('we are server')
            logging.debug('-' * 80)
            return self.fsc.get_batch(batch_size, noise=noise)
        return None

    def make_sensory_api_call(self, end_point, json_data, call_type):
        # This should implement retry...

        headers = {
            'Content-type': 'application/json',
            'API-ACCESS-KEY': self.config.API_ACCESS_KEY,
            'API-VERSION': self.config.API_VERSION
        }

        try:
            url = "%s%s:%s/%s" % (
                self.config.SENSORY_URI, self.config.SENSORY_SERVER, self.config.SENSORY_PORT, end_point)
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

    # This can be pulled from the File System Connector, doesn't need to be here, and isn't used anymore ..
    # TODO: temporary - we should calculate this using one of the provided methods this is really for testing the threaded-caching
    def get_category_map(self):
        jdata = {}
        with open('%s/category_map.json' % self.config.WEIGHTS_DIR) as data_file:
            raw_cat_data = json.load(data_file)
        for d in raw_cat_data:
            jdata[str(raw_cat_data[d])] = str(d)
        return jdata

    # TODO: temporary - we should calculate this using one of the provided methods this is really for testing the threaded-caching
    def image_sensory_store(self, data_dir, data_category, image_obj):
        filename = "%s" % datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f.png')
        path = "%s/%s/" % (data_dir, data_category)
        full_filename = "%s%s" % (path, filename)
        logging.debug("(%s)" % (full_filename))
        make_sure_path_exists(path)
        image_obj.save(full_filename)
        return True

    def sensory_batch_poll(self, batch_id):
        # lets try to grab more than one at a time // combine and return
        # since this is going to clients lets reduce chatter

        data = None
        label = None

        url = self.config.SENSORY_SERVICE_RABBITMQ_URL
        # logging.debug(url)
        parameters = pika.URLParameters(url)

        try:
            connection = pika.BlockingConnection(parameters=parameters)

            channel = connection.channel()

            method_frame, header_frame, body = channel.basic_get(batch_id)
            if method_frame:
                # logging.debug("%s %s %s" % (method_frame, header_frame, body))
                message = json.loads(body)
                label = message['label']
                data = message['data']
                # logging.debug(label)
                # logging.debug(data)
                channel.basic_ack(method_frame.delivery_tag)
            else:
                logging.debug('no message returned')
        except Exception as e:
            logging.warning(e)
        finally:
            connection.close()
        return data, label
