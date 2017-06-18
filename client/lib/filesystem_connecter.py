import os
import fnmatch
import struct
import numpy
from PIL import Image
from array import *


def get_data_set(data_directory):
  #data_directory = '/home/console/Workbench/Repositories/dicebox/train/data/'
  data_set = {}
  for root, dirnames, filenames in os.walk(data_directory):
    for filename in fnmatch.filter(filenames, '*.png'):
      new_entry = str(os.path.join(root, filename))
      new_entry = new_entry.replace(data_directory, '')
      category, filename = new_entry.split('/')
      #print("category: (%s), filename: (%s)" % (category, filename))
      data_set[new_entry] = [filename, category]
  #print(data_set)
  return data_set

def get_data_set_categories(data_set_index):
  natural_categories = []
  category_map = {}
  value_list = data_set_index.values()
  for item in value_list:
    #print(item)
    #print("natural category label: (%s)" % item[1])
    natural_categories.append(item[1])
  natural_categories = sorted(set(natural_categories))
  cat_index = 0
  for cat in natural_categories:
    #print("%i: %s" % (cat_index, cat))
    category_map[cat] = cat_index
    cat_index += 1
  #print(category_map)
  return category_map

def get_batch_list(data_set_index, batch_size):
  output = []
  set_size = len(data_set_index)
  #print("data set size: (%i)" % set_size)
  value_list = data_set_index.values()
  #print("value list: (%s)" % value_list)
  for i in range (0, batch_size):
    # get random index
    index = int(round((float(ord(struct.unpack('c', os.urandom(1))[0]))/255)*(set_size - 1)))
    #print("random index: (%i)" % index)
    output.append(value_list[index])
  #print(output)
  return output


def get_batch(data_directory, data_set_index, batch_size, category_map):
  image_data = []
  image_labels = []

  batch_list = get_batch_list(data_set_index, batch_size)

  # build file path
  for i in range(0, batch_size):
    item = batch_list[i]
    filename = "%s%s/%s" % (data_directory, item[1], item[0])
    print(filename)
    #print("  natural category label: (%s)" % item[1])
    #print("  neural network category label: (%i)" % category_map[item[1]])
    cat_one_hot = numpy.zeros(len(category_map))
    cat_one_hot[int(category_map[item[1]])] = 1
    image_labels.append(cat_one_hot)
    #print(cat_one_hot)
    pixel_data = process_image(filename)
    image_data.append(pixel_data)
  return [image_data, image_labels]


def process_image(filename):
    pixel_data = array('B')

    Im = Image.open(filename)
    pixel = Im.load()

    width, height = Im.size

    for x in range(0, width):
        for y in range(0, height):
            pixel_data.append(pixel[x, y])

    data = numpy.frombuffer(pixel_data, dtype=numpy.uint8)
    return data


#data_directory = '/home/console/Workbench/Repositories/dicebox/train/data/'
#batch_size = 20
#data_set_index = get_data_set(data_directory)
#category_map = get_data_set_categories(data_set_index)
#print(category_map)
#new_batch = get_batch(data_directory, data_set_index, batch_size, category_map)
#print(new_batch)

