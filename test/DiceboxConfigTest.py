import unittest
from src.shapeandshare.dicebox.core.config import DiceboxConfig


class DiceboxConfigTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    def setUp(self):
        self.dc = DiceboxConfig(config_file=self.local_config_file, lonestar_model_file=self.local_lonestar_model_file)

    def test_lonestar_model_v2(self):
        expected_input_size = 784
        expected_output_size = 10
        expected_optimizer = 'adamax'
        expected_dicebox_model = {
            'optimizer': expected_optimizer,
            'input_shape': [expected_input_size, ],
            'output_size': expected_output_size,
            'layers': [
                {
                    'type': 'normal',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'normal',
                    'size': 89,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'normal',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'normal',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'normal',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                }
            ]
        }

        self.assertEqual(self.dc.LONESTAR_DICEBOX_MODEL, expected_dicebox_model)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxConfigTest())
