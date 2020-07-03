import unittest
import logging
import json

from src.shapeandshare.dicebox.core import DiceboxNetwork


class DiceboxNetworkTest(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/fixtures'
    local_create_fcs = False
    local_disable_data_indexing = True
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    ACTIVATION = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
    OPTIMIZER = ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]

    def setUp(self):
        self.maxDiff = None

    def test_create_random(self):
        local_create_model = False
        local_weights_file = None

        dn = DiceboxNetwork(create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)
        self.assertEqual(dn.network, {})
        dn.create_random()
        self.assertIsNotNone(dn.network)
        logging.debug(dn.network)
        self.assertIsNot(dn.network, {})
        logging.debug(dn.network)
        dn = None

    def test_create_lonestar(self):
        local_create_model = True
        local_weights_file = None
        expected_compiled_model = None
        with open('%s/lonestar.model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        dn = DiceboxNetwork(create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        dn.create_lonestar(create_model=local_create_model, weights_filename=local_weights_file)
        returned_model = dn.model
        self.assertIsNotNone(returned_model)

        # generate a sample..
        # with open('%s/lonestar.model.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
        #     json_file.write(json.dumps(json.loads(returned_model.to_json()), indent=4))

        self.assertEqual(json.loads(returned_model.to_json()), expected_compiled_model)
        dn = None

    def test_compile_model(self):
        expected_compiled_model = None
        with open('%s/model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        local_input_size = 784
        local_output_size = 10
        local_optimizer = 'adamax'
        local_dicebox_model = {
            'optimizer': local_optimizer,
            'input_shape': [local_input_size, ],
            'output_size': local_output_size,
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

        dn = DiceboxNetwork(create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        returned_compiled_model = dn.compile_model(dicebox_model=local_dicebox_model)

        serialized_result = returned_compiled_model.to_json()

        # # generate a sample ..
        # with open('%s/model.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
        #     json_file.write(json.dumps(json.loads(serialized_result), indent=4))

        self.assertEqual(json.loads(serialized_result), expected_compiled_model)
        dn = None


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(DiceboxNetworkTest())
