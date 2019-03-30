"""Handles filesystem interactions for neural network input data.

###############################################################################
# Filesystem Connector for Neural Network Input Data
# Copyright (c) 2017-2019 Joshua Burt
###############################################################################
"""

###############################################################################
# Dependencies
###############################################################################
import os
import fnmatch
import struct
import array
import logging
import numpy
from PIL import Image
# from datetime import datetime  # used when dumping raw transforms to disk
import dicebox.docker_config


class FileSystemConnector(object):
    """File System Connector Class"""
    DATASET_INDEX = None
    DATA_DIRECTORY = None
    CATEGORY_MAP = None
    PIXEL_CACHE = {}

    CONFIG = None

    def __init__(self, data_directory, disable_data_indexing=False, config_file='./dicebox.config'):
        if self.CONFIG is None:
            self.CONFIG = dicebox.docker_config.DockerConfig(config_file)

        if self.DATA_DIRECTORY is None:
            self.DATA_DIRECTORY = os.path.normpath(data_directory)
            logging.info('data directory: (%s)', self.DATA_DIRECTORY)

        if disable_data_indexing is False:
            if self.DATASET_INDEX is None:
                self.DATASET_INDEX = self.get_data_set()
                logging.debug('DATASET_INDEX')
                logging.debug(self.DATASET_INDEX)

            if self.CATEGORY_MAP is None:
                self.CATEGORY_MAP = self.get_data_set_categories()
                logging.debug('CATEGORY_MAP')
                logging.debug(self.CATEGORY_MAP)
        else:
            logging.info('File System Connector Data Indexing Disabled.')

    def get_batch_list(self, batch_size):
        """For a given batch size, returns a random selection of indices

        :param batch_size: integer value
        :return: array of indices in the batch size (each index appearing only once).
        """
        output = []
        set_size = len(self.DATASET_INDEX)
        value_list = self.DATASET_INDEX.values()

        if batch_size > set_size:
            raise Exception('Max batch size: %s, but %s was specified!' % (set_size, batch_size))

        set_indices = []
        for i in range(0, set_size):
            set_indices.append(i)

        output_list = []
        while len(output_list) < batch_size:
            index = int(round((float(ord(struct.unpack('c', os.urandom(1))[0])) / 255) * (len(set_indices) - 1)))
            output_list.append(set_indices[index])
            set_indices.remove(set_indices[index])

        for i in output_list:
            output.append(value_list[i])
        return output

    def get_batch(self, batch_size, noise=0):
        """For a given batch size and noise level, returns a dictionary of data and labels.

        :param batch_size: integer
        :param noise: floating point number
        :return: dictionary of image data and labels
        """
        image_data = []
        image_labels = []

        category_map = self.CATEGORY_MAP
        batch_list = self.get_batch_list(batch_size)

        # build file path
        for i in range(0, batch_size):
            item = batch_list[i]
            filename = "%s%s/%s" % (self.DATA_DIRECTORY, item[1], item[0])
            logging.debug("(%s)(%s)", category_map[item[1]], filename)
            # logging.info("  natural category label: (%s)" % item[1])
            # logging.info("  neural network category label: (%i)" % category_map[item[1]])
            cat_one_hot = numpy.zeros(len(category_map))
            cat_one_hot[int(category_map[item[1]])] = 1
            image_labels.append(cat_one_hot)
            # image_labels.append(category_map[item[1]])

            # Help prevent over-fitting, and allow for new
            # sensory data to enter the cache, even when a cache
            # hit would occur.
            if self.lucky(noise):
                del self.PIXEL_CACHE[filename]

            # use pixel cache if possible
            # [k,v] (filename, pixeldata)
            if self.PIXEL_CACHE.has_key(filename):
                # found in cache
                pixel_data = self.PIXEL_CACHE[filename]
                logging.debug("loaded cached pixel data for (%s)", filename)
            else:
                pixel_data = self.process_image(filename, noise)
                self.PIXEL_CACHE[filename] = pixel_data  # add to cache
                logging.debug("cached pixel data for (%s)", filename)

            image_data.append(pixel_data)
        return [image_data, image_labels]

    @staticmethod
    def lucky(noise=0):
        """Simple function to determine whether or not a thing occurs.

        :param noise: floating point value for noise
        :return: boolean
        """
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            logging.debug('luck bestowed')
            return True
        return False

    def process_image(self, filename, noise=0):
        """For a given filename, and noise level, will return a numpy array of pixel data.

        :param filename:
        :param noise:
        :return:
        """
        pixel_data = array('B')

        im = Image.open(filename).convert('L') # Load as gray

        original_width, original_height = im.size
        # original_size = original_width, original_height
        logging.debug("original size: (%i, %i)", original_width, original_height)

        # See if there is noise
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            #  # then we introduce noise
            rotation_angle = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 * 10
            if float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 > 0.5:
                rotation_angle = rotation_angle * -1
            logging.debug("Rotating image %f", rotation_angle)
            im = im.rotate(rotation_angle)
            logging.debug("new size: (%i, %i)", im.size[0], im.size[1])

        # Perform a random scale
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            # random_scale = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            random_scale = 1.5 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            while random_scale < 0.7:
                random_scale = 1.5 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 255

            logging.debug("Scaling image %f", random_scale)
            width, height = im.size
            new_width = int(width * random_scale)
            new_height = int(height * random_scale)
            if new_width > 0 and new_height > 0:
                new_size = int(width * random_scale), int(height * random_scale)
                # logging.info(new_size)
                im = im.resize(new_size)
                # im = im.resize(new_size, Image.ANTIALIAS)
                logging.debug("new size: (%i, %i)", im.size[0], im.size[1])

        # Crop Image If Required
        logging.debug('Crop image if required')

        # Now ensure we are the same dimensions as when we started
        new_width, new_height = im.size
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

        im = im.crop((left, upper, right, lower))
        logging.debug("new size: (%i, %i)", im.size[0], im.size[1])

        # Ensure the input will match in input tensor
        #im = im.resize((original_width, original_height), Image.ANTIALIAS)
        im = im.resize((self.CONFIG.IMAGE_WIDTH, self.CONFIG.IMAGE_HEIGHT), Image.ANTIALIAS)
        logging.debug("new size: (%i, %i)", im.size[0], im.size[1])

        # dump to file for manual review
        # filename = datetime.now().strftime('transform_%Y-%m-%d_%H_%M_%S_%f.png')
        #im.save("./tmp/%s" % filename)

        pixel = im.load()

        width, height = im.size

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

    def get_data_set_categories(self):
        """Returns the dataset index as a dictionary of categories

        :return: category map as a dictionary
        """
        natural_categories = []
        category_map = {}
        value_list = self.DATASET_INDEX.values()
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

    def get_data_set(self):
        """
        Returns a dictionary of [k:filename, v:array of filename and category] for the entire data set.

        When we move this to python 3, there are much better libraries to handle this.  Checkout PurePath..
        """
        data_set = {}
        for root, dirnames, filenames in os.walk(self.DATA_DIRECTORY):
            for filename in fnmatch.filter(filenames, '*.png'):
                new_entry = str(os.path.join(root, filename))
                new_entry = new_entry.replace('%s/' % self.DATA_DIRECTORY, '')
                new_entry = os.path.normpath(new_entry)
                category, filename = new_entry.split('/')
                # logging.info("category: (%s), filename: (%s)" % (category, filename))
                data_set[new_entry] = [filename, category]
        # logging.info(data_set)
        return data_set
