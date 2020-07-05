import json
import unittest
from src.shapeandshare.dicebox.core.config import DiceboxConfig


class DiceboxConfigTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE

    def setUp(self):
        self.dc = DiceboxConfig(config_file=self.local_config_file)

    def test_config(self):
        self.assertEqual(self.dc.NOISE, 0.0)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxConfigTest())
