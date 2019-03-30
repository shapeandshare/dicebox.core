import unittest
import dicebox.filesystem_connecter as filesystemconnectorclass
import logging
import sys
import json


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
    DATASET_LOCATION = 'test/data/test_dataset/data'
    DISABLE_DATA_INDEXING = False

    def setUp(self):
        # instantiate the file system connector Class
        self.fsc = filesystemconnectorclass.FileSystemConnector(self.DATASET_LOCATION, self.DISABLE_DATA_INDEXING)

        with open('test/data/DATASET_INDEX.json') as json_file:
            self.EXPECTED_DATASET_INDEX = json.load(json_file)
        if self.EXPECTED_DATASET_INDEX is None:
            Exception('Unable to load data/DATASET_INDEX.json!')

        with open('test/data/CATEGORY_MAP.json') as json_file:
            self.EXPECTED_CATEGORY_MAP = json.load(json_file)
        if self.EXPECTED_CATEGORY_MAP is None:
            Exception('Unable to load data/CATEGORY_MAP.json!')

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


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
