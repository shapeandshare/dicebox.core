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

    def setUp(self):
        local_create_fcs = False
        local_disable_data_indexing = True
        local_config_file='./dicebox.config'

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

        self.dn = DiceboxNetwork(nn_param_choices=local_nn_param_choices,
                                 create_fcs=local_create_fcs,
                                 disable_data_indexing=local_disable_data_indexing,
                                 config_file=local_config_file)

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

        returned_compiled_model = self.dn.compile_model(network=local_network,
                                                        nb_classes=local_nbclasses,
                                                        input_shape=local_input_shape)
        # with open('model.txt', 'w') as f:
        #     f.write(returned_compiled_model.to_json())
        serialized_result = returned_compiled_model.to_json()
        self.assertEqual(json.loads(serialized_result), expected_compiled_model)


    def test_multi_layer_size(self):
        expected_compiled_model = None
        with open('%s/multi_model.json' % self.TEST_DATA_BASE) as json_file:
            expected_compiled_model = json.load(json_file)
        self.assertIsNotNone(expected_compiled_model)

        local_dicebox_model = [
            {
                'type': 'normal',
                'size': 987,
                'activation': 'elu',
                'input_shape': [784,]
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
        local_input_size = 784
        local_output_size = 10
        local_optimizer = 'adamax'
        returned_compiled_model = self.dn.compile_model_v2(dicebox_model=local_dicebox_model,
                                                           input_shape=local_input_size,
                                                           output_size=local_output_size,
                                                           optimizer=local_optimizer)
        # with open('multi_model.json', 'w') as f:
        #     f.write(returned_compiled_model.to_json())

        serialized_result = returned_compiled_model.to_json()
        self.assertEqual(json.loads(serialized_result), expected_compiled_model)



if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()
