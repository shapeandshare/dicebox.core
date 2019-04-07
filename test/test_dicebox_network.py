import unittest
import logging
import sys
import json
import numpy
import numpy.testing
from dicebox.dicebox_network import DiceboxNetwork


root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


class Test(unittest.TestCase):
    """
    The basic class that inherits unittest.TestCase
    """
    TEST_DATA_BASE = 'test/data'
    local_create_fcs = False
    local_disable_data_indexing = True
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE
    local_lonestar_model_file = '%s/dicebox.lonestar.json' % TEST_DATA_BASE

    NB_NEURONS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    NB_LAYERS = [1, 2, 3, 5, 8, 13, 21]
    ACTIVATION = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
    OPTIMIZER = ["rmsprop", "adam", "sgd", "adagrad", "adadelta", "adamax", "nadam"]
    local_nn_param_choices = {
        'nb_neurons': NB_NEURONS,
        'nb_layers': NB_LAYERS,
        'activation': ACTIVATION,
        'optimizer': OPTIMIZER
    }

#    def setUp(self):


    def test_compile_model(self):
        expected_compiled_model = None
        with open('%s/model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        local_network = {}
        local_network['nb_layers'] = 5
        local_network['activation'] = 'elu'
        local_network['optimizer'] = 'adamax'
        local_network['nb_neurons'] = 987
        local_nbclasses = 10  # The number of categories
        local_input_shape = [784, ]  # length x width (28x28=784)

        dn = DiceboxNetwork(nn_param_choices=self.local_nn_param_choices,
                            create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        returned_compiled_model = dn.compile_model(network=local_network,
                                                        nb_classes=local_nbclasses,
                                                        input_shape=local_input_shape)
        # with open('model.txt', 'w') as f:
        #     f.write(returned_compiled_model.to_json())
        serialized_result = returned_compiled_model.to_json()
        self.assertEqual(json.loads(serialized_result), expected_compiled_model)
        dn = None

    def test_compile_model_v2(self):
        expected_compiled_model = None
        with open('%s/model_v2.json' % self.TEST_DATA_BASE) as json_file:
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

        dn = DiceboxNetwork(nn_param_choices=self.local_nn_param_choices,
                            create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        returned_compiled_model = dn.compile_model_v2(dicebox_model=local_dicebox_model)
        # with open('model_v2.json', 'w') as f:
        #     f.write(returned_compiled_model.to_json())

        serialized_result = returned_compiled_model.to_json()
        self.assertEqual(json.loads(serialized_result), expected_compiled_model)
        dn = None

    def test_create_lonestar(self):
        local_create_model = True
        local_weights_file = None
        expected_compiled_model = None
        with open('%s/lonestar.model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        dn = DiceboxNetwork(nn_param_choices=self.local_nn_param_choices,
                            create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        dn.create_lonestar(create_model=local_create_model, weights_filename=local_weights_file)
        returned_model = dn.model
        self.assertIsNotNone(returned_model)

        # with open('%s/lonestar.model.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
        #     json_file.write(json.dumps(json.loads(returned_model.to_json()), indent=4))
        self.assertEqual(json.loads(returned_model.to_json()), expected_compiled_model)
        dn = None

    def test_create_lonestar_v2(self):
        local_create_model = True
        local_weights_file = None
        expected_compiled_model = None
        with open('%s/lonestar.model_v2.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        dn = DiceboxNetwork(nn_param_choices=self.local_nn_param_choices,
                            create_fcs=self.local_create_fcs,
                            disable_data_indexing=self.local_disable_data_indexing,
                            config_file=self.local_config_file,
                            lonestar_model_file=self.local_lonestar_model_file)

        dn.create_lonestar_v2(create_model=local_create_model, weights_filename=local_weights_file)
        returned_model = dn.model_v2
        self.assertIsNotNone(returned_model)

        # with open('%s/lonestar.model_v2.out.json' % self.TEST_DATA_BASE, 'w') as json_file:
        #     json_file.write(json.dumps(json.loads(returned_model.to_json()), indent=4))
        self.assertEqual(json.loads(returned_model.to_json()), expected_compiled_model)
        dn = None


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
