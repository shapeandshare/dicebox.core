from abc import ABC
from enum import Enum
from typing import List, Union

from .layer import DenseLayer, DropoutLayer, DenseLayerConfigure, DropoutLayerConfigure
from ..config import DiceboxConfig
from ..layer_factory import LayerFactory


class Optimizers(Enum):
    RMSPROP = 1
    ADAM = 2
    SGD = 3
    ADAGRAD = 4
    ADADELTA = 5
    ADAMAX = 6
    NADAM = 7


class NetworkConfig:
    def __init__(self, input_shape: int, output_size: int, optimizer: Optimizers):
        self.input_shape: int = input_shape
        self.output_size: int = output_size
        self.optimizer: Optimizers = optimizer


class Network(ABC):
    config: DiceboxConfig
    layer_factory: LayerFactory

    def __init__(self, config: DiceboxConfig, network_config: NetworkConfig):
        self.config = config
        self.input_shape: int = network_config.input_shape
        self.output_size: int = network_config.output_size
        self.optimizer: Optimizers = network_config.optimizer
        self.layers: List[Union[DropoutLayer, DenseLayer]] = []

    def add_layer(self, layer_config: Union[DropoutLayerConfigure, DenseLayerConfigure]):
        self.layers.append(LayerFactory(self.config).build_layer(layer_config=layer_config))
