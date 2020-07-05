from abc import ABC, abstractmethod
from enum import Enum
from typing import Union


class LayerType(Enum):
    DROPOUT = 1
    DENSE = 2


class ActivationFunction(Enum):
    SOFTMAX = 1
    ELU = 2
    SOFTPLUS = 3
    SOFTSIGN = 4
    RELU = 5
    TANH = 6
    SIGMOID = 7
    HARD_SIGMOID = 8
    LINEAR = 9


class LayerConfig(ABC):
    layer_type: LayerType

    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type

    def type(self) -> LayerType:
        return self.layer_type


class DropoutLayerConfigure(LayerConfig):
    def __init__(self, rate: float):
        super().__init__(LayerType.DROPOUT)
        self.rate: float = rate


class DenseLayerConfigure(LayerConfig):
    def __init__(self, size: int, activation: ActivationFunction):
        super().__init__(LayerType.DENSE)
        self.size: int = size
        self.activation: ActivationFunction = activation


class Layer(ABC):
    layer_type: LayerType

    def __init__(self, layer_type: LayerType):
        self.layer_type: LayerType = layer_type

    def type(self) -> LayerType:
        return self.layer_type


class DenseLayer(Layer):
    def __init__(self, config: DenseLayerConfigure):
        super().__init__(layer_type=config.type())
        self.size: int = config.size
        self.activation: ActivationFunction = config.activation


class DropoutLayer(Layer):
    def __init__(self, config: DropoutLayerConfigure):
        super().__init__(layer_type=config.type())
        self.type: LayerType = config.type()
        self.rate: float = config.rate
