from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.layers.core import Dropout

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

    # Processes the __network which we store externally
    def create_network(self, network_definition) -> Network:
        optimizer: Optimizers = network_definition['optimizer'].upper()
        input_shape: int = network_definition['input_shape']
        output_size: int = network_definition['output_size']

        new_network_config = NetworkConfig(input_shape=input_shape, output_size=output_size, optimizer=optimizer)
        new_network = Network(self.config, new_network_config)

        # Process layers
        for layer in network_definition['layers']:
            if layer['type'] == 'dense':
                size: int = layer['size']
                activation: ActivationFunction = ActivationFunction[layer['activation'].upper()]
                new_layer_config = self.layer_factory.build_dense_layer_config(size=size, activation=activation)
                new_network.add_layer(new_layer_config)
            elif layer['type'] == 'dropout':
                rate: float = layer['rate']
                new_layer_config = self.layer_factory.build_dropout_layer_config(rate=rate)
                new_network.add_layer(new_layer_config)
            else:
                raise

        return new_network

    def create_random_network(self) -> Network:
        # Select an optimizer
        optimizer_index: int = random_index(len(self.config.TAXONOMY['optimizer']))
        optimizer: str = self.config.TAXONOMY['optimizer'][optimizer_index - 1]

        network_config: NetworkConfig = NetworkConfig(input_shape=self.config.INPUT_SHAPE,
                                                      output_size=self.config.NB_CLASSES,
                                                      optimizer=Optimizers[optimizer.upper()])
        network: Network = Network(self.config, network_config)

        # Determine the number of layers..
        layer_count: int = random_index_between(self.config.TAXONOMY['min_layers'],
                                                self.config.TAXONOMY['max_layers'])
        for layer_index in range(1, layer_count):
            # add new random layer to the __network
            network.add_layer(self.layer_factory.build_random_layer_config())

        return network

    @staticmethod
    def compile_network(dicebox_network: Network) -> Sequential:
        model = Sequential()

        first_layer: bool = False
        for layer in dicebox_network.layers:
            # build and add layer
            if layer.type() == LayerType.DROPOUT:
                # handle dropout
                model.add(Dropout(layer.rate))
            else:
                neurons: int = layer.size
                activation: ActivationFunction = layer.activation

                if first_layer is False:
                    first_layer = True
                    model.add(Dense(neurons, activation=activation, input_shape=dicebox_network.input_shape))
                else:
                    model.add(Dense(neurons, activation=activation))

        # add final output layer.
        model.add(Dense(dicebox_network.output_size,
                        activation='softmax'))  # TODO: Make it possible to define with from the enum...

        model.compile(loss='categorical_crossentropy', optimizer=dicebox_network.optimizer, metrics=['accuracy'])

        return model
