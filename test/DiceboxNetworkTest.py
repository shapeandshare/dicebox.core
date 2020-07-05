import unittest
import logging
import json
from typing import Any

from src.shapeandshare.dicebox.core import DiceboxNetwork
from src.shapeandshare.dicebox.core.config import DiceboxConfig
from src.shapeandshare.dicebox.core.models.network import Network
from src.shapeandshare.dicebox.core.network_factory import NetworkFactory


class DiceboxNetworkTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    # local_create_fcs = True
    # local_disable_data_indexing = True
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    # ACTIVATION = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
    # OPTIMIZER = ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]

    def setUp(self):
        self.maxDiff = None

    def test_create_random(self):
        dc: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
        nf: NetworkFactory = NetworkFactory(config=dc)
        nf.create_random_network()

        # dn: DiceboxNetwork = DiceboxNetwork(dc,
        #                     create_fsc=True,
        #                     disable_data_indexing=True)
        # dn.generate_random_network()

        # self.assertEqual(dn.__network, {})
        # dn.__network = dn.__network_factory.create_random_network()
        # self.assertIsNotNone(dn.__network)
        # logging.debug(dn.__network)
        # self.assertIsNot(dn.__network, {})
        # logging.debug(dn.__network)
        # dn = None

    def test_load_network(self):
        dc = DiceboxConfig(config_file=self.local_config_file)
        expected_dicebox_serialized_model = json.load(open(self.local_lonestar_model_file))

        expected_compiled_model: Any = None
        with open('%s/lonestar.model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)

        local_input_size = 784
        local_output_size = 10
        local_optimizer = 'adamax'
        local_network_definition = {
            'optimizer': local_optimizer,
            'input_shape': [local_input_size, ],
            'output_size': local_output_size,
            'layers': [
                {
                    'type': 'dense',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'dense',
                    'size': 89,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'dense',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'dense',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                },
                {
                    'type': 'dense',
                    'size': 987,
                    'activation': 'elu'
                },
                {
                    'type': 'dropout',
                    'rate': 0.2
                }
            ]
        }

        nf = NetworkFactory(config=dc)
        dn = nf.create_network(network_definition=local_network_definition)

        # dn.__network_factory.create_network(network_definition=)
        # dn.create_lonestar(create_model=local_create_model, weights_filename=local_weights_file)
        # returned_model = dn.__model
        # self.assertIsNotNone(returned_model)

        # generate a sample..
        # with open('%s/lonestar.__model.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
        #     json_file.write(json.dumps(json.loads(returned_model.to_json()), indent=4))

        # self.assertEqual(json.loads(returned_model.to_json()), expected_compiled_model)
        # dn = None

    # def test_compile_model(self):
    #     expected_compiled_model = None
    #     with open('%s/__model.json' % self.TEST_DATA_BASE) as json_file:
    #         expected_compiled_model = json.load(json_file)
    #     self.assertIsNotNone(expected_compiled_model)
    #
    #     local_input_size = 784
    #     local_output_size = 10
    #     local_optimizer = 'adamax'
    #     local_dicebox_model_definition = {
    #         'optimizer': local_optimizer,
    #         'input_shape': [local_input_size, ],
    #         'output_size': local_output_size,
    #         'layers': [
    #             {
    #                 'type': 'normal',
    #                 'size': 987,
    #                 'activation': 'elu'
    #             },
    #             {
    #                 'type': 'dropout',
    #                 'rate': 0.2
    #             },
    #             {
    #                 'type': 'normal',
    #                 'size': 89,
    #                 'activation': 'elu'
    #             },
    #             {
    #                 'type': 'dropout',
    #                 'rate': 0.2
    #             },
    #             {
    #                 'type': 'normal',
    #                 'size': 987,
    #                 'activation': 'elu'
    #             },
    #             {
    #                 'type': 'dropout',
    #                 'rate': 0.2
    #             },
    #             {
    #                 'type': 'normal',
    #                 'size': 987,
    #                 'activation': 'elu'
    #             },
    #             {
    #                 'type': 'dropout',
    #                 'rate': 0.2
    #             },
    #             {
    #                 'type': 'normal',
    #                 'size': 987,
    #                 'activation': 'elu'
    #             },
    #             {
    #                 'type': 'dropout',
    #                 'rate': 0.2
    #             }
    #         ]
    #     }
    #
    #     dn = DiceboxNetwork(create_fcs=self.local_create_fcs,
    #                         disable_data_indexing=self.local_disable_data_indexing,
    #                         config_file=self.local_config_file,
    #                         lonestar_model_file=self.local_lonestar_model_file)
    #
    #     local_network: Network = dn.__network_factory.create_network(local_dicebox_model_definition)
    #     returned_compiled_model = dn.__network_factory.compile_network(dicebox_network=local_network)
    #
    #     serialized_result = returned_compiled_model.to_json()
    #
    #     # # generate a sample ..
    #     # with open('%s/__model.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
    #     #     json_file.write(json.dumps(json.loads(serialized_result), indent=4))
    #
    #     self.assertEqual(json.loads(serialized_result), expected_compiled_model)
    #     dn = None


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxNetworkTest())
