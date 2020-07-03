import unittest
import src.dicebox.utils.helpers as helpers

class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """

    def test_lucky(self):
        noise = 0
        self.assertFalse(helpers.lucky(noise))

        noise = 1
        self.assertTrue(helpers.lucky(noise))


