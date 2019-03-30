import unittest
import dicebox.filesystem_connecter as filesystemconnectorclass
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

dataset_location = 'test/data/test_dataset/data'
disable_data_indexing = False

class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    # instantiate the file system connector Class
    fsc = filesystemconnectorclass.FileSystemConnector(dataset_location, disable_data_indexing)

if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
