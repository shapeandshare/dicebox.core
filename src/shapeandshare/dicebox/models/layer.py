from abc import ABC
from enum import Enum
from random import choices

from typing import Tuple


class LayerType(Enum):
    DROPOUT = "dropout"
    DENSE = "dense"
    CONV2D = "conv2d"  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D


def select_random_layer_type() -> LayerType:
    return choices([LayerType.DROPOUT, LayerType.DENSE, LayerType.CONV2D])[0]


class ActivationFunction(Enum):
    ELU = "elu"
    EXPONENTIAL = "exponential"
    HARD_SIGMOID = "hard_sigmoid"
    LINEAR = "linear"
    RELU = "relu"
    SELU = "selu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    SOFTSIGN = "softsign"
    SWISH = "swish"
    TANH = "tanh"


class Conv2DPadding(Enum):
    VALID = "valid"
    SAME = "same"


def select_random_conv2d_padding_type() -> Conv2DPadding:
    return choices([Conv2DPadding.VALID, Conv2DPadding.SAME])[0]


class Layer(ABC):
    layer_type: LayerType

    def __init__(self, layer_type: LayerType):
        self.layer_type: LayerType = layer_type


class DenseLayer(Layer):
    size: int
    activation: ActivationFunction

    def __init__(self, size: int, activation: ActivationFunction):
        super().__init__(layer_type=LayerType.DENSE)
        self.size = size
        self.activation = activation


class DropoutLayer(Layer):
    rate: float

    def __init__(self, rate: float):
        super().__init__(layer_type=LayerType.DROPOUT)
        self.rate = rate


# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
class Conv2DLayer(Layer):
    filters: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    padding: Conv2DPadding
    activation: ActivationFunction

    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Conv2DPadding,
        activation: ActivationFunction,
    ):
        super().__init__(layer_type=LayerType.CONV2D)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
