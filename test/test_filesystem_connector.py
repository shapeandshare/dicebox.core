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

    DATASET_LOCATION = 'test/data/test_dataset/data'
    DISABLE_DATA_INDEXING = False

    def setUp(self):
        # instantiate the file system connector Class
        self.fsc = filesystemconnectorclass.FileSystemConnector(self.DATASET_LOCATION, self.DISABLE_DATA_INDEXING)

    def test_class_variable_DATA_DIRECTORY(self):
        self.assertEqual(self.DATASET_LOCATION, self.fsc.DATA_DIRECTORY)

    def test_class_variable_DATASET_INDEX(self):
        expected_dataset_index = None
        with open('test/data/DATASET_INDEX.json') as json_file:
            expected_dataset_index = json.load(json_file)
        if expected_dataset_index is None:
            Exception('Unable to load data/DATASET_INDEX.json!')

        self.assertEqual(expected_dataset_index, self.fsc.DATASET_INDEX)

    def test_class_variable_CATEGORY_MAP(self):
        expected_category_map = None
        with open('test/data/CATEGORY_MAP.json') as json_file:
            expected_category_map = json.load(json_file)
        if expected_category_map is None:
            Exception('Unable to load data/CATEGORY_MAP.json!')

        self.assertEqual(expected_category_map, self.fsc.CATEGORY_MAP)




if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
