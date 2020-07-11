import unittest
import json
from typing import Any

from src.config.dicebox_config import DiceboxConfig
from src.factories.network_factory import NetworkFactory
from src.models.network import Network


class NetworkFactoryTest(unittest.TestCase):
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE

    dicebox_config: DiceboxConfig = DiceboxConfig(config_file=local_config_file)
    network_factory: NetworkFactory = NetworkFactory(config=dicebox_config)

    def setUp(self):
        self.maxDiff = None

    def test_create_random_network(self):
        new_network_one: Network = self.network_factory.create_random_network()
        new_network_two: Network = self.network_factory.create_random_network()
        self.assertNotEqual(new_network_one.decompile(), new_network_two.decompile())

    def test_create_network(self):
        new_network_one: Network = self.network_factory.create_random_network()
        decompiled_network_one: Any = new_network_one.decompile()
        new_network_two: Network = self.network_factory.create_network(decompiled_network_one)
        decompiled_network_two: Any = new_network_two.decompile()
        self.assertEqual(decompiled_network_one, decompiled_network_two)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(NetworkFactoryTest())
