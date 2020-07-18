from typing import Any, Tuple

from .layer_factory import LayerFactory
from ..config.dicebox_config import DiceboxConfig
from ..models.layer import LayerType, ActivationFunction, Conv2DLayer, Conv2DPadding, DenseLayer, DropoutLayer
from ..models.network import Network, Optimizers
from ..utils.helpers import random_index, random_index_between


# The birthing chambers ...

class NetworkFactory(LayerFactory):
    def __init__(self, config: DiceboxConfig):
        super().__init__(config=config)

    # Processes the network which we store externally
    def create_network(self, network_definition: Any) -> Network:
        optimizer: Optimizers = Optimizers(network_definition['optimizer'])
        input_shape: int = network_definition['input_shape']
        output_size: int = network_definition['output_size']

        new_network = Network(config=self.config, input_shape=input_shape, output_size=output_size, optimizer=optimizer)

        if 'layers' not in network_definition:
            network_definition['layers'] = []

        # Process layers
        for layer in network_definition['layers']:
            if layer['type'] == LayerType.DENSE.value:
                size: int = layer['size']
                activation: ActivationFunction = ActivationFunction(layer['activation'])
                new_layer: DenseLayer = self.build_dense_layer(size=size, activation=activation)
                new_network.add_layer(new_layer)
            elif layer['type'] == LayerType.DROPOUT.value:
                rate: float = layer['rate']
                new_layer: DropoutLayer = self.build_dropout_layer(rate=rate)
                new_network.add_layer(new_layer)
            elif layer['type'] == LayerType.CONV2D.value:
                kernel_size: Tuple[int, int] = layer['kernel_size']
                strides: Tuple[int, int] = layer['strides']
                padding: Conv2DPadding = layer['padding']
                activation: ActivationFunction = layer['activation']
                new_layer: Conv2DLayer = Conv2DLayer(kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)
                new_network.add_layer(new_layer)
            else:
                raise Exception("Unsupported layer type: (%s) provided." % layer['type'])

        new_network.compile()
        return new_network

    def create_random_network(self) -> Network:
        # Select an optimizer
        optimizer_index: int = random_index(len(self.config.TAXONOMY['optimizer']))
        optimizer: str = self.config.TAXONOMY['optimizer'][optimizer_index - 1]

        network: Network = Network(config=self.config,
                                   input_shape=self.config.INPUT_SHAPE,
                                   output_size=self.config.NB_CLASSES,
                                   optimizer=Optimizers(optimizer))

        # Determine the number of layers..
        layer_count: int = random_index_between(self.config.TAXONOMY['min_layers'],
                                                self.config.TAXONOMY['max_layers'])
        for layer_index in range(0, layer_count):
            # add new random layer to the network
            network.add_layer(self.build_random_layer())

        network.compile()
        return network
