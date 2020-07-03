import unittest
import logging
import sys


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def setUp(self):
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)

    def test_logging(self):
        logging.info('setup logger')


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(Test())
