from abc import ABC
from enum import Enum


class LayerType(Enum):
    DROPOUT = 'dropout'
    DENSE = 'dense'


class ActivationFunction(Enum):
    SOFTMAX = 'softmax'
    ELU = 'elu'
    SOFTPLUS = 'softplus'
    SOFTSIGN = 'softsign'
    RELU = 'relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    HARD_SIGMOID = 'hard_sigmoid'
    LINEAR = 'linear'


class Layer(ABC):
    layer_type: LayerType

    def __init__(self, layer_type: LayerType):
        self.layer_type: LayerType = layer_type


class DenseLayer(Layer):
    def __init__(self, size: int, activation: ActivationFunction):
        super().__init__(layer_type=LayerType.DENSE)
        self.size: int = size
        self.activation: ActivationFunction = activation


class DropoutLayer(Layer):
    def __init__(self, rate: float):
        super().__init__(layer_type=LayerType.DROPOUT)
        self.rate: float = rate
