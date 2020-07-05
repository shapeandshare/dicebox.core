from typing import Union

from .config import DiceboxConfig
from .models.layer import DenseLayerConfigure, DropoutLayerConfigure, ActivationFunction, LayerType, DropoutLayer, \
    DenseLayer
from .utils import random_index, random_index_between, random_strict


class LayerFactory:
    config: DiceboxConfig = None

    def __init__(self, config: DiceboxConfig):
        self.config = config

    def build_random_layer_config(self) -> Union[DropoutLayerConfigure, DenseLayerConfigure]:
        # determine what the layer type will be
        layer_type_index = random_index(len(self.config.TAXONOMY['layer_types']))
        layer_type = self.config.TAXONOMY['layer_types'][layer_type_index - 1]

        if layer_type == LayerType.DROPOUT.value:
            return LayerFactory.build_dropout_layer_config(rate=random_strict())
        elif layer_type == LayerType.DENSE.value:
            # determine the size and activation function to use.
            size: int = random_index_between(self.config.TAXONOMY['min_neurons'],
                                             self.config.TAXONOMY['max_neurons'])
            activation_index: int = random_index(len(self.config.TAXONOMY['activation']))
            activation: str = self.config.TAXONOMY['activation'][activation_index - 1]

            return LayerFactory.build_dense_layer_config(size=size, activation=ActivationFunction(activation))
        else:
            raise

    @staticmethod
    def build_dropout_layer_config(rate: float) -> DropoutLayerConfigure:
        return DropoutLayerConfigure(rate=rate)

    @staticmethod
    def build_dense_layer_config(size: int, activation: ActivationFunction) -> DenseLayerConfigure:
        return DenseLayerConfigure(size=size, activation=activation)

    @staticmethod
    def compile_layer(layer_config: Union[DenseLayerConfigure, DropoutLayerConfigure]) -> Union[
        DenseLayer, DropoutLayer]:
        if layer_config.layer_type == LayerType.DROPOUT:
            return DropoutLayer(layer_config)
        elif layer_config.layer_type == LayerType.DENSE:
            return DenseLayer(layer_config)
        else:
            raise

    @staticmethod
    def decompile_layer(layer: Union[DenseLayer, DropoutLayer]) -> Union[DenseLayerConfigure, DropoutLayerConfigure]:
        if layer.layer_type == LayerType.DENSE.value:
            return DenseLayerConfigure(size=layer.size, activation=layer.activation)
        elif layer.layer_type == LayerType.DROPOUT.value:
            return DropoutLayerConfigure(rate=layer.rate)
        else:
            raise
