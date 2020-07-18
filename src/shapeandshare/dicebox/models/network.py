from typing import List, Union, Any, Tuple

from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, Flatten
from tensorflow.python.keras.models import Sequential

from .layer import DenseLayer, DropoutLayer, Conv2DLayer, LayerType, ActivationFunction, Conv2DPadding
from .optimizers import Optimizers
from ..config.dicebox_config import DiceboxConfig
from ..factories.layer_factory import LayerFactory


class Network(LayerFactory):
    # genomotype
    __input_shape: Tuple[int, int, int]
    __output_size: int
    __optimizer: Optimizers
    __layers: List[Union[DenseLayer, DropoutLayer, Conv2DLayer]]

    # phenotype
    model: Union[Sequential, None]

    def __init__(self, config: DiceboxConfig, input_shape: Tuple[int, int, int], output_size: int, optimizer: Optimizers, layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = None):
        super().__init__(config=config)
        self.__input_shape: Tuple[int, int, int] = input_shape
        self.__output_size: int = output_size
        self.__optimizer: Optimizers = optimizer
        if layers is not None:
            self.__layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = layers
        else:
            self.__layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = []
        self.model: Union[Sequential, None] = None
        self.compile()

    def add_layer(self, layer: Union[DropoutLayer, DenseLayer, Conv2DLayer]) -> None:
        self.__layers.append(layer)
        self.compile()

    def __clear_model(self):
        if self.model:
            del self.model
        self.model = None

    def compile(self) -> None:
        self.__clear_model()

        # early exist for empty layers
        if len(self.__layers) < 1:
            return

        model = Sequential()

        first_layer: bool = True
        for layer in self.__layers:
            # build and add layer
            if layer.layer_type == LayerType.DROPOUT:
                # handle dropout
                model.add(Dropout(layer.rate))
            elif layer.layer_type == LayerType.DENSE:
                neurons: int = layer.size
                activation: ActivationFunction = layer.activation
                if first_layer is True:
                    first_layer = False
                    model.add(Dense(neurons, activation=activation.value, input_shape=self.__input_shape))
                else:
                    model.add(Dense(neurons, activation=activation.value))
            elif layer.layer_type == LayerType.CONV2D:
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
                kernel_size: Tuple[int, int] = layer.kernel_size
                strides: Tuple[int, int] = layer.strides
                filters: int = layer.filters
                padding: Conv2DPadding = layer.padding
                activation: ActivationFunction = layer.activation
                print("filters=%i, kernel_size=%s, strides=%s, padding=%s, activation=%s" % (filters, kernel_size, strides, padding.value, activation.value))
                if first_layer is True:
                    first_layer = False
                    # model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding.value, activation=activation.value, input_shape=self.__input_shape))
                    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation.value, input_shape=self.__input_shape))
                else:
                    model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation.value))
                    # model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding.value, activation=activation.value))
            else:
                raise
        # add final output layer.
        model.add(Dense(self.__output_size,
                        activation='softmax'))  # TODO: Make it possible to define with from the enum...
        model.compile(loss='categorical_crossentropy', optimizer=self.__optimizer.value, metrics=['accuracy'])

        # return model
        self.model = model

    def get_layer(self, layer_index: int) -> Union[DenseLayer, DropoutLayer, Conv2DLayer]:
        return self.__layers[layer_index]

    def get_layers(self) -> List[Union[DenseLayer, DropoutLayer, Conv2DLayer]]:
        return self.__layers

    def get_optimizer(self) -> Optimizers:
        return self.__optimizer

    def get_input_shape(self) -> int:
        return self.__input_shape

    def get_output_size(self) -> int:
        return self.__output_size

    def decompile(self) -> Any:
        definition = {
            'input_shape': self.__input_shape,
            'output_size': self.__output_size,
            'optimizer': self.__optimizer.value,
            'layers': []
        }

        for i in range(0, len(self.__layers)):
            layer = self.decompile_layer(self.__layers[i])
            definition['layers'].append(layer)

        return definition

