import unittest
from src.dicebox.config.dicebox_config import DiceboxConfig


class DiceboxConfigTest(unittest.TestCase):
    fixtures_base = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % fixtures_base

    def setUp(self):
        self.dc = DiceboxConfig(config_file=self.local_config_file)

    def test_config(self):
        self.assertEqual(self.dc.NOISE, 0.0)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxConfigTest())
