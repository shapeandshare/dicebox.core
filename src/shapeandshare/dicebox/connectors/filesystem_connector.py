import array
import fnmatch
import logging
import os
import struct
from typing import Dict, Any, List, Union

import numpy
from PIL import Image
from numpy import ndarray

from ..config.dicebox_config import DiceboxConfig
from ..utils.helpers import lucky


class FileSystemConnector:
    """File System Connector Class"""

    config: DiceboxConfig = None

    dataset_index: dict
    category_map: Dict[Any, int]
    pixel_cache = {}

    def __init__(self, config: DiceboxConfig, data_directory: str, disable_data_indexing=False):
        self.config = config

        self.data_directory = os.path.normpath(data_directory)
        logging.info('data directory: (%s)', self.data_directory)

        if disable_data_indexing is False:

            self.dataset_index: dict = self.get_data_set()
            logging.debug('dataset_index')
            logging.debug(self.dataset_index)

            self.category_map: Dict[Any, int] = self.get_data_set_categories()
            logging.debug('CATEGORY_MAP')
            logging.debug(self.category_map)
        else:
            logging.info('File System Connector Data Indexing Disabled.')

    def get_batch_list(self, batch_size: int) -> List[int]:
        """For a given batch size, returns a random selection of indices

        :param batch_size: integer value
        :return: array of indices in the batch size (each index appearing only once).
        """
        output: List[int] = []
        set_size: int = len(self.dataset_index)
        value_list = list(self.dataset_index.values())
        if batch_size > set_size:
            raise Exception('Max batch size: %s, but %s was specified!' % (set_size, batch_size))

        set_indices: List[int] = []
        for i in range(0, set_size):
            set_indices.append(i)

        output_list: List[int] = []
        while len(output_list) < batch_size:
            index: int = int(round((float(ord(struct.unpack('c', os.urandom(1))[0])) / 255) * (len(set_indices) - 1)))
            output_list.append(set_indices[index])
            set_indices.remove(set_indices[index])

        for i in output_list:
            output.append(value_list[i])
        return output

    def get_batch(self, batch_size: int, noise: float = 0.0) -> List[Union[list, List[ndarray]]]:
        """For a given batch size and noise level, returns a dictionary of data and labels.

        :param batch_size: integer
        :param noise: floating point number
        :return: dictionary of image data and labels
        """
        image_data = []
        image_labels = []

        category_map: Dict[Any, int] = self.category_map
        batch_list: [] = self.get_batch_list(batch_size)

        # build file path
        for i in range(0, batch_size):
            item = batch_list[i]
            filename = "%s/%s/%s" % (self.data_directory, item[1], item[0])
            logging.debug("(%s)(%s)", category_map[item[1]], filename)
            # logging.info("  natural category label: (%s)" % item[1])
            # logging.info("  neural __network category label: (%i)" % category_map[item[1]])
            cat_one_hot = numpy.zeros(len(category_map))
            cat_one_hot[int(category_map[item[1]])] = 1
            image_labels.append(cat_one_hot)
            # image_labels.append(category_map[item[1]])

            # Help prevent over-fitting, and allow for new
            # sensory data to enter the cache, even when a cache
            # hit would occur.
            if filename in self.pixel_cache and not lucky(noise):
                del self.pixel_cache[filename]

            # use pixel cache if possible
            # [k,v] (filename, pixeldata)
            if filename in self.pixel_cache:
                # found in cache
                pixel_data = self.pixel_cache[filename]
                logging.debug("loaded cached pixel data for (%s)", filename)
            else:
                pixel_data = self.process_image(filename, noise)
                self.pixel_cache[filename] = pixel_data  # add to cache
                logging.debug("cached pixel data for (%s)", filename)

            image_data.append(pixel_data)
        return [image_data, image_labels]

    def process_image(self, filename: str, noise: float = 0.0):
        """For a given filename, and noise level, will return a numpy array of pixel data.

        :param filename:
        :param noise:
        :return:
        """
        pixel_data = array.array('B')

        local_image = Image.open(filename).convert('L')  # Load as gray

        original_width, original_height = local_image.size
        # original_size = original_width, original_height
        logging.debug("original size: (%i, %i)", original_width, original_height)

        # See if there is noise
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            #  # then we introduce noise
            rotation_angle = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 * 10
            if float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 > 0.5:
                rotation_angle = rotation_angle * -1
            logging.debug("Rotating image %f", rotation_angle)
            local_image = local_image.rotate(rotation_angle)
            logging.debug("new size: (%i, %i)", local_image.size[0], local_image.size[1])

        # Perform a random scale
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            # random_scale = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            random_scale = 1.5 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            while random_scale < 0.7:
                random_scale = 1.5 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 255

            logging.debug("Scaling image %f", random_scale)
            width, height = local_image.size
            new_width = int(width * random_scale)
            new_height = int(height * random_scale)
            if new_width > 0 and new_height > 0:
                new_size = int(width * random_scale), int(height * random_scale)
                # logging.info(new_size)
                local_image = local_image.resize(new_size)
                # local_image = local_image.resize(new_size, Image.ANTIALIAS)
                logging.debug("new size: (%i, %i)", local_image.size[0], local_image.size[1])

        # Crop Image If Required
        logging.debug('Crop image if required')

        # Now ensure we are the same dimensions as when we started
        new_width, new_height = local_image.size
        # if new_width > original_width or new_height > original_height:
        new_middle_x = float(new_width) / 2
        new_middle_y = float(new_height) / 2
        left = int(new_middle_x - float(original_width) / 2)
        upper = int(new_middle_y - float(original_height) / 2)
        right = int(new_middle_x + float(original_width) / 2)
        lower = int(new_middle_y + float(original_height) / 2)
        logging.debug("left: %i", left)
        logging.debug("upper: %i", upper)
        logging.debug("right: %i", right)
        logging.debug("lower: %i", lower)

        local_image = local_image.crop((left, upper, right, lower))
        logging.debug("new size: (%i, %i)", local_image.size[0], local_image.size[1])

        # Ensure the input will match in input tensor
        # local_image = local_image.resize((original_width, original_height), Image.ANTIALIAS)
        local_image = local_image.resize((self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT), Image.ANTIALIAS)
        logging.debug("new size: (%i, %i)", local_image.size[0], local_image.size[1])

        # dump to file for manual review
        # filename = datetime.now().strftime('transform_%Y-%m-%d_%H_%M_%S_%f.png')
        # local_image.save("./tmp/%s" % filename)

        pixel = local_image.load()

        width, height = local_image.size

        for x in range(0, width):
            for y in range(0, height):
                pixel_value = pixel[x, y]
                # logging.info("(%f)" % float(noise))
                # chance = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
                # logging.info("(%f)" % chance)
                # if float(noise) >= float(chance):
                #  # logging.info("  adding pixel noise (%f/%f)" % (float(noise), chance)) # add noise
                #  pixel_value = int(ord(struct.unpack('c', os.urandom(1))[0]))
                pixel_data.append(pixel_value)

        data = numpy.frombuffer(pixel_data, dtype=numpy.uint8)
        return data

    def get_data_set_categories(self) -> Dict[Any, int]:
        """Returns the dataset index as a dictionary of categories

        :return: category map as a dictionary
        """
        natural_categories = []
        category_map = {}
        value_list = self.dataset_index.values()
        # print(self.dataset_index)
        # print(value_list)
        for item in value_list:
            # logging.info(item)
            # logging.info("natural category label: (%s)" % item[1])
            natural_categories.append(item[1])
        natural_categories = sorted(set(natural_categories))
        cat_index = 0
        for cat in natural_categories:
            # logging.info("%i: %s" % (cat_index, cat))
            category_map[cat] = cat_index
            cat_index += 1
        # logging.info(category_map)
        return category_map

    def get_data_set(self) -> dict:
        """
        Returns a dictionary of [k:filename, v:array of filename and category] for the entire data set.

        When we move this to python 3, there are much better libraries to handle this.  Checkout PurePath..
        """
        data_set: Dict[str, List[Union[str, Dict[str, Any]]]] = {}
        category: Dict[str, Any]
        # filename: str
        for root, _, filenames in os.walk(self.data_directory):
            for filename in fnmatch.filter(filenames, '*.png'):
                new_entry = str(os.path.join(root, filename))
                new_entry = new_entry.replace('%s%s' % (self.data_directory, os.path.sep), '')
                new_entry = os.path.normpath(new_entry)
                category, filename = new_entry.split(os.path.sep)
                # logging.info("category: (%s), filename: (%s)" % (category, filename))
                data_set[new_entry]: Union[str, Dict[str, Any]] = [filename, category]
        # logging.info(data_set)
        return data_set
