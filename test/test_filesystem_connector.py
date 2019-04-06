import unittest
import dicebox.filesystem_connecter as filesystemconnectorclass
import logging
import sys
import json
import numpy
import numpy.testing

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)



class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    EXPECTED_DATASET_INDEX = None
    EXPECTED_CATEGORY_MAP = None
    TEST_DATA_BASE = 'test/data'
    DATASET_LOCATION = '%s/test_dataset/data' % TEST_DATA_BASE
    DICEBOX_CONFIG_FILE = '%s/dicebox.config' % TEST_DATA_BASE
    DISABLE_DATA_INDEXING = False

    def setUp(self):
        # instantiate the file system connector Class
        self.fsc = filesystemconnectorclass.FileSystemConnector(self.DATASET_LOCATION, self.DISABLE_DATA_INDEXING, self.DICEBOX_CONFIG_FILE)

        with open('%s/DATASET_INDEX.json' % self.TEST_DATA_BASE) as json_file:
            self.EXPECTED_DATASET_INDEX = json.load(json_file)
        if self.EXPECTED_DATASET_INDEX is None:
            Exception('Unable to load %s/DATASET_INDEX.json!', self.TEST_DATA_BASE)

        with open('%s/CATEGORY_MAP.json' % self.TEST_DATA_BASE) as json_file:
            self.EXPECTED_CATEGORY_MAP = json.load(json_file)
        if self.EXPECTED_CATEGORY_MAP is None:
            Exception('Unable to load %s/CATEGORY_MAP.json!', self.TEST_DATA_BASE)

    def test_class_variable_DATA_DIRECTORY(self):
        self.assertEqual(self.DATASET_LOCATION, self.fsc.DATA_DIRECTORY)

    def test_class_variable_DATASET_INDEX(self):
        self.assertEqual(self.EXPECTED_DATASET_INDEX, self.fsc.DATASET_INDEX)

    def test_class_variable_CATEGORY_MAP(self):
        self.assertEqual(self.EXPECTED_CATEGORY_MAP, self.fsc.CATEGORY_MAP)

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
            self.assertEqual(e.message, 'Max batch size: 2, but 3 was specified!')

    def test_lucky(self):
        noise = 0
        self.assertFalse(self.fsc.lucky(noise))

        noise = 1
        self.assertTrue(self.fsc.lucky(noise))

    def test_process_image(self):
        filename = '%s/0/mnist_testing_0_28x28_3.png' % self.DATASET_LOCATION

        noise = 0
        expected_data = numpy.fromfile('%s/0/mnist_testing_0_28x28_3.png.nbarray.binary' % self.DATASET_LOCATION, dtype=numpy.uint8)
        returned_data = self.fsc.process_image(filename, noise)
        # returned_data.tofile('test/data/test_dataset/data/0/mnist_testing_0_28x28_3.png.nbarray.binary')
        numpy.testing.assert_array_equal(returned_data, expected_data)

    def test_get_data_set_categories(self):
        returned_categories = self.fsc.get_data_set_categories()
        self.assertEqual(returned_categories, self.EXPECTED_CATEGORY_MAP)

    def test_get_data_set(self):
        returned_data_set = self.fsc.get_data_set()
        self.assertEqual(returned_data_set, self.EXPECTED_DATASET_INDEX)



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
