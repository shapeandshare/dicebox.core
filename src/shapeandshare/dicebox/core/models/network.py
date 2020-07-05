from abc import ABC
from enum import Enum
from typing import List, Union

from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.models import Sequential

from .layer import DenseLayer, DropoutLayer, DenseLayerConfigure, DropoutLayerConfigure, LayerType, ActivationFunction
from ..config import DiceboxConfig
from ..layer_factory import LayerFactory


class Optimizers(Enum):
    RMSPROP = 'rmsprop'
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    ADADELTA = 'adadelta'
    ADAMAX = 'adamax'
    NADAM = 'nadam'


class NetworkConfig:
    def __init__(self, input_shape: int, output_size: int, optimizer: Optimizers):
        self.input_shape: int = input_shape
        self.output_size: int = output_size
        self.optimizer: Optimizers = optimizer


class Network(ABC):
    def __init__(self, config: DiceboxConfig, network_config: NetworkConfig):
        self.config: DiceboxConfig = config
        self.input_shape: int = network_config.input_shape
        self.output_size: int = network_config.output_size
        self.optimizer: Optimizers = network_config.optimizer

        self.layer_factory: LayerFactory = LayerFactory(self.config)
        self.layers: List[Union[DropoutLayer, DenseLayer]] = []

    def add_layer(self, layer_config: Union[DropoutLayerConfigure, DenseLayerConfigure]) -> None:
        self.layers.append(self.layer_factory.compile_layer(layer_config=layer_config))

    def compile(self) -> Sequential:
        model = Sequential()

        first_layer: bool = True
        for layer in self.layers:
            # build and add layer
            if layer.layer_type == LayerType.DROPOUT.value:
                # handle dropout
                model.add(Dropout(layer.rate))
            elif layer.layer_type == LayerType.DENSE.value:
                neurons: int = layer.size
                activation: ActivationFunction = layer.activation
                if first_layer is True:
                    first_layer = False
                    model.add(Dense(neurons, activation=activation.value, input_shape=self.input_shape))
                else:
                    model.add(Dense(neurons, activation=activation.value))
            else:
                raise
        # add final output layer.
        model.add(Dense(self.output_size,
                        activation='softmax'))  # TODO: Make it possible to define with from the enum...
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer.value, metrics=['accuracy'])

        return model

    # def compile_layer(self, layer_config: Union[DenseLayerConfigure, DropoutLayerConfigure]):


    def decompile_layer(self, layer: Union[DenseLayer, DropoutLayer]) -> Union[DenseLayerConfigure, DropoutLayerConfigure]:
        return self.layer_factory.decompile_layer(layer)

    def get_layer_definition(self, layer_index: int):
        layer: Union[DenseLayer, DropoutLayer] = self.layers[layer_index]
        config: Union[DenseLayerConfigure, DropoutLayerConfigure] = self.decompile_layer(layer)
