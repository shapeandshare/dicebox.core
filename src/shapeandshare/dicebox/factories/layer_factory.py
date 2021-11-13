from abc import ABC
from typing import Union, Any, Tuple

from ..config.dicebox_config import DiceboxConfig
from ..models.layer import (
    ActivationFunction,
    LayerType,
    DropoutLayer,
    DenseLayer,
    Conv2DLayer,
    Conv2DPadding,
    select_random_conv2d_padding_type, FlattenLayer,
)
from ..utils.helpers import random_index, random_index_between, random_strict


class LayerFactory(ABC):
    config: DiceboxConfig = None

    def __init__(self, config: DiceboxConfig):
        self.config = config

    def build_random_layer(self) -> Union[DropoutLayer, DenseLayer, Conv2DLayer, FlattenLayer]:
        # determine what the layer type will be
        layer_type_index = random_index_between(min_index=0, max_index=len(self.config.TAXONOMY["layer_types"]))
        # layer_type_index = random_index(len(self.config.TAXONOMY['layer_types']))
        layer_type = self.config.TAXONOMY["layer_types"][layer_type_index - 1]
        if layer_type == LayerType.FLATTEN.value:
            return LayerFactory.build_flatten_layer()
        elif layer_type == LayerType.DROPOUT.value:
            return LayerFactory.build_dropout_layer(rate=random_strict())
        elif layer_type == LayerType.DENSE.value:
            # determine the size and activation function to use.
            size: int = random_index_between(self.config.TAXONOMY["min_neurons"], self.config.TAXONOMY["max_neurons"])
            activation_index: int = random_index(len(self.config.TAXONOMY["activation"]))
            activation: str = self.config.TAXONOMY["activation"][activation_index - 1]

            return LayerFactory.build_dense_layer(size=size, activation=ActivationFunction(activation))
        elif layer_type == LayerType.CONV2D.value:
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
            filters: int = random_index_between(1, self.config.MAX_NEURONS)  # TODO?
            kernel_size: Tuple[int, int] = (random_index_between(1, 4), random_index_between(1, 4))
            strides: Tuple[int, int] = (1, 1)  # (random_index_between(1, 2), random_index_between(1, 2))

            activation_index: int = random_index(len(self.config.TAXONOMY["activation"]))
            activation: str = self.config.TAXONOMY["activation"][activation_index - 1]

            return LayerFactory.build_conv2d_layer(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=select_random_conv2d_padding_type(),
                activation=ActivationFunction(activation),
            )
        else:
            raise Exception("Unsupported layer type: (%s) provided." % layer_type)

    @staticmethod
    def build_flatten_layer() -> FlattenLayer:
        return FlattenLayer()

    @staticmethod
    def build_dropout_layer(rate: float) -> DropoutLayer:
        return DropoutLayer(rate=rate)

    @staticmethod
    def build_dense_layer(size: int, activation: ActivationFunction) -> DenseLayer:
        return DenseLayer(size=size, activation=activation)

    @staticmethod
    def build_conv2d_layer(
        filters: int,
        kernel_size: Tuple[int, int],
        strides: Tuple[int, int],
        padding: Conv2DPadding,
        activation: ActivationFunction,
    ) -> Conv2DLayer:
        return Conv2DLayer(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation
        )

    @staticmethod
    def decompile_layer(layer: Union[DenseLayer, DropoutLayer, Conv2DLayer]) -> Any:
        definition = {}

        if layer.layer_type == LayerType.DROPOUT:
            definition["type"] = LayerType.DROPOUT.value
            definition["rate"] = layer.rate
        elif layer.layer_type == LayerType.FLATTEN:
            definition["type"] = LayerType.FLATTEN.value
        elif layer.layer_type == LayerType.DENSE:
            definition["type"] = LayerType.DENSE.value
            definition["size"] = layer.size
            definition["activation"] = layer.activation.value
        elif layer.layer_type == LayerType.CONV2D:
            definition["type"] = LayerType.CONV2D.value
            definition["filters"] = layer.filters
            definition["kernel_size"] = layer.kernel_size
            definition["strides"] = layer.strides
            definition["padding"] = layer.padding.value
            definition["activation"] = layer.activation.value
        else:
            raise

        return definition
