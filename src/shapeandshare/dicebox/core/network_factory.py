from typing import Any

from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential

from .config import DiceboxConfig
from .layer_factory import LayerFactory
from .models.layer import LayerType, ActivationFunction
from .models.network import Network, NetworkConfig, Optimizers
from .utils import random_index, random_index_between


# The birthing chambers ...

class NetworkFactory:
    config: DiceboxConfig
    layer_factory: LayerFactory

    # TODO: lonestar should not exist, make it the responsiblity of the caller during create
    def __init__(self, config: DiceboxConfig):
        self.config = config
        self.layer_factory = LayerFactory(config=self.config)

    # Processes the network which we store externally
    def create_network(self, network_definition: Any) -> Network:
        optimizer: Optimizers = Optimizers(network_definition['optimizer'])
        input_shape: int = network_definition['input_shape']
        output_size: int = network_definition['output_size']

        new_network_config = NetworkConfig(input_shape=input_shape, output_size=output_size, optimizer=optimizer)
        new_network = Network(self.config, new_network_config)

        # Process layers
        for layer in network_definition['layers']:
            if layer['type'] == LayerType.DENSE.value:
                size: int = layer['size']
                activation: ActivationFunction = ActivationFunction(layer['activation'])
                new_layer_config = self.layer_factory.build_dense_layer_config(size=size, activation=activation)
                new_network.add_layer(new_layer_config)
            elif layer['type'] == LayerType.DROPOUT.value:
                rate: float = layer['rate']
                new_layer_config = self.layer_factory.build_dropout_layer_config(rate=rate)
                new_network.add_layer(new_layer_config)
            else:
                raise

        new_network.compile()
        return new_network

    def create_random_network(self) -> Network:
        # Select an optimizer
        optimizer_index: int = random_index(len(self.config.TAXONOMY['optimizer']))
        optimizer: str = self.config.TAXONOMY['optimizer'][optimizer_index - 1]

        network_config: NetworkConfig = NetworkConfig(input_shape=self.config.INPUT_SHAPE,
                                                      output_size=self.config.NB_CLASSES,
                                                      optimizer=Optimizers(optimizer))
        network: Network = Network(self.config, network_config)

        # Determine the number of layers..
        layer_count: int = random_index_between(self.config.TAXONOMY['min_layers'],
                                                self.config.TAXONOMY['max_layers'])
        for layer_index in range(1, layer_count):
            # add new random layer to the network
            network.add_layer(self.layer_factory.build_random_layer_config())

        network.compile()
        return network

