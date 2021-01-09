import os
import unittest
from typing import Any

from src.shapeandshare.dicebox.config.dicebox_config import DiceboxConfig
from src.shapeandshare.dicebox.factories.network_factory import NetworkFactory
from src.shapeandshare.dicebox.models.network import Network
from src.shapeandshare.dicebox.models.optimizers import Optimizers


class NetworkFactoryTest(unittest.TestCase):
    TEST_DATA_BASE = "test/fixtures"
    local_config_file = "%s/dicebox.config" % TEST_DATA_BASE

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

    def test_should_throw_exception_when_asked_to_create_an_unknown_layer_type(self):
        os.environ["LAYER_TYPES"] = '["random", "unsupported"]'
        local_dicebox_config: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
        local_network_factory: NetworkFactory = NetworkFactory(config=local_dicebox_config)

        definition = {
            "input_shape": 1,
            "output_size": 1,
            "optimizer": Optimizers.ADAM.value,
            "layers": [{"type": "random"}],
        }

        try:
            local_network_factory.create_network(network_definition=definition)
            self.assertTrue(False, "Expected exception not seen.")
        except Exception:
            self.assertTrue(True, "Expected exception seen.")

        del os.environ["LAYER_TYPES"]


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(NetworkFactoryTest())
