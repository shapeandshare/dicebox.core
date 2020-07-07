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


class LayerConfig(ABC):
    layer_type: LayerType

    def __init__(self, layer_type: LayerType):
        self.layer_type = layer_type


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
        super().__init__(layer_type=config.layer_type)
        self.size: int = config.size
        self.activation: ActivationFunction = config.activation


class DropoutLayer(Layer):
    def __init__(self, config: DropoutLayerConfigure):
        super().__init__(layer_type=config.layer_type)
        self.type: LayerType = config.layer_type
        self.rate: float = config.rate
