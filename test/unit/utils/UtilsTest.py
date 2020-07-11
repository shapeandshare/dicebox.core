import unittest

from src.shapeandshare.dicebox.utils.helpers import lucky


class UtilsTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def test_lucky(self):
        noise = 0
        self.assertFalse(lucky(noise))

        noise = 1
        self.assertTrue(lucky(noise))


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(UtilsTest())
