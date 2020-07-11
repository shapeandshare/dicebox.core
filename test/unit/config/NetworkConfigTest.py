import unittest
from typing import Union, List

from src.config.network_config import NetworkConfig
from src.models.layer import DropoutLayer, DenseLayer
from src.models.optimizers import Optimizers


class NetworkConfigTest(unittest.TestCase):
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE

    def setUp(self):
        self.maxDiff = None

    def test_create_with_layers(self):
        layer: DropoutLayer = DropoutLayer(rate=0.0)
        layers: List[Union[DropoutLayer, DenseLayer]] = [layer]
        network_config: NetworkConfig = NetworkConfig(input_shape=1, output_size=1, optimizer=Optimizers.ADAM, layers=layers)

        self.assertEqual(network_config.input_shape, 1)
        self.assertEqual(network_config.output_size, 1)
        self.assertEqual(network_config.optimizer, Optimizers.ADAM)
        self.assertEqual(network_config.layers, layers)
        self.assertEqual(network_config.model, None)

    def test_create_without_layers(self):
        network_config: NetworkConfig = NetworkConfig(input_shape=1, output_size=1, optimizer=Optimizers.ADAM)

        self.assertEqual(network_config.input_shape, 1)
        self.assertEqual(network_config.output_size, 1)
        self.assertEqual(network_config.optimizer, Optimizers.ADAM)
        self.assertEqual(network_config.layers, None)
        self.assertEqual(network_config.model, None)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(NetworkConfigTest())
