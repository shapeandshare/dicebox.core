import pickle
import unittest
import logging
import json
import numpy
import numpy.testing
from PIL.Image import Image

from src.shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from src.shapeandshare.dicebox.connectors.filesystem_connector import FileSystemConnector

from tensorflow.keras.preprocessing.image import array_to_img


class FileSystemConnectorTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    dc: DiceboxConfig
    fsc: FileSystemConnector

    EXPECTED_DATASET_INDEX = None
    EXPECTED_CATEGORY_MAP = None
    TEST_DATA_BASE = 'test/fixtures'
    DATASET_LOCATION = '%s/test_dataset/data' % TEST_DATA_BASE
    DICEBOX_CONFIG_FILE = '%s/dicebox.config' % TEST_DATA_BASE
    LONESTAR_MODEL_FILE = '%s/dicebox.lonestar.json' % TEST_DATA_BASE
    DISABLE_DATA_INDEXING = False

    def setUp(self):
        self.dc = DiceboxConfig(config_file=self.DICEBOX_CONFIG_FILE)

        # instantiate the file system connector Class
        self.fsc = FileSystemConnector(data_directory=self.DATASET_LOCATION,
                                       config=self.dc,
                                       disable_data_indexing=self.DISABLE_DATA_INDEXING)

        with open('%s/DATASET_INDEX.json' % self.TEST_DATA_BASE) as json_file:
            self.EXPECTED_DATASET_INDEX = json.load(json_file)
        if self.EXPECTED_DATASET_INDEX is None:
            Exception('Unable to load %s/DATASET_INDEX.json', self.TEST_DATA_BASE)

        with open('%s/CATEGORY_MAP.json' % self.TEST_DATA_BASE) as json_file:
            self.EXPECTED_CATEGORY_MAP = json.load(json_file)
        if self.EXPECTED_CATEGORY_MAP is None:
            Exception('Unable to load %s/CATEGORY_MAP.json', self.TEST_DATA_BASE)

    def test_class_variable_DATA_DIRECTORY(self):
        self.assertEqual(self.DATASET_LOCATION, self.fsc.data_directory)

    def test_class_variable_DATASET_INDEX(self):
        self.assertEqual(self.EXPECTED_DATASET_INDEX, self.fsc.dataset_index)

    def test_class_variable_CATEGORY_MAP(self):
        self.assertEqual(self.EXPECTED_CATEGORY_MAP, self.fsc.category_map)

    def test_get_batch_list(self):
        batch_size = 0
        expected_batch = []
        returned_batch = self.fsc.get_batch_list(batch_size)
        self.assertEqual(expected_batch, returned_batch)

        batch_size = 1
        returned_batch = self.fsc.get_batch_list(batch_size)
        logging.debug(returned_batch[0])
        item = returned_batch[0]
        found = False
        for value in self.EXPECTED_DATASET_INDEX.values():
            if value == item:
                found = True
                break
        self.assertTrue(found)

        batch_size = 2
        returned_batch = self.fsc.get_batch_list(batch_size)
        logging.debug(returned_batch)

        batch_size = 3
        try:
            returned_batch = self.fsc.get_batch_list(batch_size)
        except Exception as e:
            self.assertEqual(str(e), 'Max batch size: 2, but 3 was specified!')

    def test_get_batch(self):
        returned_batch = self.fsc.get_batch(2, 0)

    def test_process_image(self):
        filename = '%s/0/mnist_testing_0_28x28_3.png' % self.DATASET_LOCATION
        noise = 0
        returned_data = self.fsc.process_image(filename, noise)

        # with open("test/fixtures/test_dataset/data/0/mnist_testing_0_28x28_3.png.pickle", "wb") as file:
            # Pickle the 'data' dictionary using the highest protocol available.
            # pickle.dump(returned_data, file, pickle.HIGHEST_PROTOCOL)

        with open("%s/0/mnist_testing_0_28x28_3.png.pickle" % self.DATASET_LOCATION, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            expected_data = pickle.load(f)

        # local_image: Image = array_to_img(returned_data)
        # local_image.save('test_output.png', format='png')

        numpy.testing.assert_array_equal(returned_data, expected_data)

    def test_get_data_set_categories(self):
        returned_categories = self.fsc.get_data_set_categories()
        self.assertEqual(returned_categories, self.EXPECTED_CATEGORY_MAP)

    def test_get_data_set(self):
        returned_data_set = self.fsc.get_data_set()
        self.assertEqual(returned_data_set, self.EXPECTED_DATASET_INDEX)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(FileSystemConnectorTest())
