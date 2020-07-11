import os
import unittest
from typing import Any, Union

from src.config.dicebox_config import DiceboxConfig
from src.factories.layer_factory import LayerFactory
from src.models.layer import DropoutLayer, ActivationFunction, DenseLayer


class LayerFactoryTest(unittest.TestCase):
    TEST_DATA_BASE = 'test/fixtures'
    local_config_file = '%s/dicebox.config' % TEST_DATA_BASE

    dicebox_config: DiceboxConfig = DiceboxConfig(config_file=local_config_file)
    layer_factory: LayerFactory = LayerFactory(config=dicebox_config)

    def setUp(self):
        self.maxDiff = None

    def test_layer_dropout_generation(self):
        dropout_layer: DropoutLayer = self.layer_factory.build_dropout_layer(rate=0.0)
        decompiled_dropout_layer: Any = self.layer_factory.decompile_layer(dropout_layer)
        self.assertEqual(decompiled_dropout_layer, {'type': 'dropout', 'rate': 0.0})

    def test_layer_dense_generation(self):
        dense_layer: DenseLayer = self.layer_factory.build_dense_layer(size=0, activation=ActivationFunction.SOFTMAX)
        decompiled_dense_layer: Any = self.layer_factory.decompile_layer(dense_layer)
        self.assertEqual(decompiled_dense_layer, {'activation': 'softmax', 'size': 0, 'type': 'dense'})

    def test_build_random_layer(self):
        layer_one: Union[DropoutLayer, DenseLayer] = self.layer_factory.build_random_layer()
        layer_two: Union[DropoutLayer, DenseLayer] = self.layer_factory.build_random_layer()
        decompiled_layer_one = self.layer_factory.decompile_layer(layer_one)
        decompiled_layer_two = self.layer_factory.decompile_layer(layer_two)
        self.assertNotEqual(decompiled_layer_one, decompiled_layer_two)

    def test_decompile_layer(self):
        layer_one: DropoutLayer = DropoutLayer(rate=1.0)
        layer_two: DenseLayer = DenseLayer(size=1, activation=ActivationFunction.SOFTMAX)
        decompiled_layer_one = self.layer_factory.decompile_layer(layer_one)
        decompiled_layer_two = self.layer_factory.decompile_layer(layer_two)
        self.assertEqual(decompiled_layer_one, {'type': 'dropout', 'rate': 1.0})
        self.assertEqual(decompiled_layer_two, {'type': 'dense', 'size': 1, 'activation': 'softmax'})

    def test_should_throw_exception_for_unknown_layer_type(self):
        os.environ['LAYER_TYPES'] = '["random", "unsupported"]'
        local_dicebox_config: DiceboxConfig = DiceboxConfig(config_file=self.local_config_file)
        local_layer_factory: LayerFactory = LayerFactory(config=local_dicebox_config)
        del os.environ['LAYER_TYPES']
        try:
            local_layer_factory.build_random_layer()
            self.assertFalse(True, 'Exception should have been thrown.')
        except Exception:
            self.assertTrue(True, 'Expected exception seen.')

    def test_should_throw_exception_when_decompiling_unknown_layer_type(self):
        bad_layer: DropoutLayer = DropoutLayer(rate=0.0)
        bad_layer.layer_type = 'BLAH'
        try:
            self.layer_factory.decompile_layer(bad_layer)
            self.assertFalse(True, 'Exception should have been thrown.')
        except Exception:
            self.assertTrue(True, 'Expected exception seen.')


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(LayerFactoryTest())
