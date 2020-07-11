import unittest

from src.models.layer import DenseLayer, ActivationFunction, LayerType, DropoutLayer


class LayerTest(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def test_dense_layer(self):
        dense = DenseLayer(size=1, activation=ActivationFunction.SOFTMAX)
        self.assertEqual(dense.layer_type, LayerType.DENSE)
        self.assertEqual(dense.size, 1)
        self.assertEqual(dense.activation, ActivationFunction.SOFTMAX)

    def test_dropout_layer(self):
        dropout = DropoutLayer(rate=0.0)
        self.assertEqual(dropout.layer_type, LayerType.DROPOUT)
        self.assertEqual(dropout.rate, 0.0)


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(LayerTest())
