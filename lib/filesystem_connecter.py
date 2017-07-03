import os
import fnmatch
import struct
import numpy
from PIL import Image
from array import *
import logging
from datetime import datetime


class FileSystemConnector():

    DATASET_INDEX = None
    DATA_DIRECTORY = None
    CATEGORY_MAP = None
    PIXEL_CACHE = {}

    def __init__(self, data_directory):
        if FileSystemConnector.DATA_DIRECTORY is None:
            FileSystemConnector.DATA_DIRECTORY = data_directory

        if FileSystemConnector.DATASET_INDEX is None:
            FileSystemConnector.DATASET_INDEX = FileSystemConnector.get_data_set(self)
            logging.debug('DATASET_INDEX')
            logging.debug(FileSystemConnector.DATASET_INDEX)

        if FileSystemConnector.CATEGORY_MAP is None:
            FileSystemConnector.CATEGORY_MAP = FileSystemConnector.get_data_set_categories(self)
            logging.debug('CATEGORY_MAP')
            logging.debug(FileSystemConnector.CATEGORY_MAP)


    def get_batch_list(self, batch_size):
        output = []
        set_size = len(FileSystemConnector.DATASET_INDEX)
        value_list = FileSystemConnector.DATASET_INDEX.values()

        set_indices = []
        for i in range(1, set_size):
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
        image_data = []
        image_labels = []

        category_map = FileSystemConnector.CATEGORY_MAP
        batch_list = self.get_batch_list(batch_size)

        # build file path
        for i in range(0, batch_size):
            item = batch_list[i]
            filename = "%s%s/%s" % (FileSystemConnector.DATA_DIRECTORY, item[1], item[0])
            logging.debug("(%s)(%s)" % (category_map[item[1]], filename))
            # logging.info("  natural category label: (%s)" % item[1])
            # logging.info("  neural network category label: (%i)" % category_map[item[1]])
            cat_one_hot = numpy.zeros(len(category_map))
            cat_one_hot[int(category_map[item[1]])] = 1
            image_labels.append(cat_one_hot)
            # image_labels.append(category_map[item[1]])
            # logging.info(cat_one_hot)

            # use pixel cache if possible
            # [k,v] (filename, pixeldata)
            if FileSystemConnector.PIXEL_CACHE.has_key(filename) and self.lucky(noise):
                # and self.lucky(noise):
                # found in cache
                pixel_data = FileSystemConnector.PIXEL_CACHE[filename]
                logging.debug("loaded cached pixel data for (%s)" % filename)
            else:
                pixel_data = self.process_image(filename, noise)
                FileSystemConnector.PIXEL_CACHE[filename] = pixel_data # add to cache
                logging.debug("cached pixel data for (%s)" % filename)

            image_data.append(pixel_data)
        return [image_data, image_labels]

    def lucky(self, noise=0.0):
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            logging.debug('luck bestowed')
            return True
        return False

    def process_image(self, filename, noise=0):
        pixel_data = array('B')

        Im = Image.open(filename)

        original_width, original_height = Im.size
        # original_size = original_width, original_height
        logging.debug("original size: (%i, %i)" % (original_width, original_height))

        ## See if there is noise
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            #  # then we introduce noise
            rotation_angle = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 * 25
            if float(ord(struct.unpack('c', os.urandom(1))[0])) / 255 > 0.5:
                rotation_angle = rotation_angle * -1
            logging.debug("Rotating image %f" % rotation_angle)
            Im = Im.rotate(rotation_angle)
            logging.debug("new size: (%i, %i)" % (Im.size[0], Im.size[1]))

        # Perform a random scale
        if float(noise) > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
            # random_scale = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            random_scale = 2 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
            while (random_scale < 0.9):
                random_scale = 2 * float(ord(struct.unpack('c', os.urandom(1))[0])) / 500

            logging.debug("Scaling image %f" % random_scale)
            width, height = Im.size
            new_width = int(width * random_scale)
            new_height = int(height * random_scale)
            if new_width > 0 and new_height > 0:
                new_size = int(width * random_scale), int(height * random_scale)
                # logging.info(new_size)
                Im = Im.resize(new_size)
                # Im = Im.resize(new_size, Image.ANTIALIAS)
                logging.debug("new size: (%i, %i)" % (Im.size[0], Im.size[1]))

        # Crop Image If Required
        logging.debug('Crop image if required')
        # Now ensure we are the same dimensions as when we started
        new_width, new_height = Im.size
        # if new_width > original_width or new_height > original_height:
        new_middle_x = float(new_width) / 2
        new_middle_y = float(new_height) / 2
        left = int(new_middle_x - float(original_width) / 2)
        upper = int(new_middle_y - float(original_height) / 2)
        right = int(new_middle_x + float(original_width) / 2)
        lower = int(new_middle_y + float(original_height) / 2)
        logging.debug("left: %i" % left)
        logging.debug("upper: %i" % upper)
        logging.debug("right: %i" % right)
        logging.debug("lower: %i" % lower)

        Im = Im.crop((left, upper, right, lower))
        logging.debug("new size: (%i, %i)" % (Im.size[0], Im.size[1]))
        Im = Im.resize((original_width, original_height), Image.ANTIALIAS)
        logging.debug("new size: (%i, %i)" % (Im.size[0], Im.size[1]))

        # dump to file for manual review
        # filename = datetime.now().strftime('transform_%Y-%m-%d_%H_%M_%S_%f.png')
        # Im.save("./tmp/%s" % filename)

        pixel = Im.load()

        width, height = Im.size

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

    # public method
    def get_data_set_categories(self):
        natural_categories = []
        category_map = {}
        value_list = FileSystemConnector.DATASET_INDEX.values()
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
        # data_directory = '/home/console/Workbench/Repositories/dicebox/train/data/'
        #data_directory = FileSystemConnector.DATA_DIRECTORY
        data_set = {}
        for root, dirnames, filenames in os.walk(FileSystemConnector.DATA_DIRECTORY):
            for filename in fnmatch.filter(filenames, '*.png'):
                new_entry = str(os.path.join(root, filename))
                new_entry = new_entry.replace(FileSystemConnector.DATA_DIRECTORY, '')
                category, filename = new_entry.split('/')
                # logging.info("category: (%s), filename: (%s)" % (category, filename))
                data_set[new_entry] = [filename, category]
        # logging.info(data_set)
        return data_set