from enum import Enum
from typing import List, Union

from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.models import Sequential

from .layer import DenseLayer, DropoutLayer, DenseLayerConfigure, DropoutLayerConfigure, LayerType, ActivationFunction
from ..config.dicebox_config import DiceboxConfig
from ..factories.layer_factory import LayerFactory


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


class Network(LayerFactory):
    def __init__(self, config: DiceboxConfig, network_config: NetworkConfig):
        super().__init__(config=config)
        self.input_shape: int = network_config.input_shape
        self.output_size: int = network_config.output_size
        self.optimizer: Optimizers = network_config.optimizer

        self.layers: List[Union[DropoutLayer, DenseLayer]] = []
        self.model: Union[Sequential, None] = None

    def add_layer(self, layer_config: Union[DropoutLayerConfigure, DenseLayerConfigure]) -> None:
        self.layers.append(self.compile_layer(layer_config=layer_config))

    def clear_model(self):
        if self.model:
            del self.model
        self.model: Union[Sequential, None] = None

    def compile(self) -> None:
        self.clear_model()
        model = Sequential()

        first_layer: bool = True
        for layer in self.layers:
            # build and add layer
            if layer.layer_type == LayerType.DROPOUT:
                # handle dropout
                model.add(Dropout(layer.rate))
            elif layer.layer_type == LayerType.DENSE:
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

        # return model
        self.model = model

    def get_layer_definition(self, layer_index: int) -> Union[DenseLayer, DropoutLayer]:
        return self.layers[layer_index]

    def get_layer(self, layer_index: int) -> Union[DenseLayer, DropoutLayer]:
        return self.layers[layer_index]
