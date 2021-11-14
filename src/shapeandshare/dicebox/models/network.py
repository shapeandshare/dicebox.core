import logging
from typing import List, Union, Any, Tuple

from tensorflow.python.keras.layers import Dropout, Dense, Conv2D, Flatten
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.models import Sequential

from .layer import DenseLayer, DropoutLayer, Conv2DLayer, LayerType, ActivationFunction, Conv2DPadding, FlattenLayer
from .optimizers import Optimizers
from ..config.dicebox_config import DiceboxConfig
from ..factories.layer_factory import LayerFactory


class Network(LayerFactory):
    # genomotype (not included in the superclass via the dice)
    __optimizer: Optimizers
    __layers: List[Union[DenseLayer, DropoutLayer, Conv2DLayer]]

    # phenotype
    model: Union[Sequential, None]

    def __init__(
        self,
        config: DiceboxConfig,
        optimizer: Optimizers,
        layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = None,
    ):
        super().__init__(config=config)
        self.__optimizer: Optimizers = optimizer
        if layers is not None:
            self.__layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = layers
        else:
            self.__layers: List[Union[DropoutLayer, DenseLayer, Conv2DLayer]] = []
        self.model: Union[Sequential, None] = None
        self.compile()

    def add_layer(self, layer: Union[DropoutLayer, DenseLayer, Conv2DLayer, FlattenLayer]) -> None:
        self.__layers.append(layer)
        self.compile()

    def __clear_model(self):
        if self.model:
            del self.model
        self.model = None

    # Not thread safe
    def compile(self) -> None:
        self.__clear_model()

        # early exit for empty layers
        if len(self.__layers) < 1:
            return

        model = Sequential()

        first_layer: bool = True
        for layer in self.__layers:
            # build and add layer
            if layer.layer_type == LayerType.FLATTEN:
                # flatten
                if first_layer is True:
                    model.add(Flatten(input_shape=self.config.INPUT_SHAPE))
                    first_layer = False
                else:
                    model.add(Flatten())
            elif layer.layer_type == LayerType.DROPOUT:
                # handle dropout
                model.add(Dropout(rate=layer.rate))
                if first_layer:
                    first_layer = False
            elif layer.layer_type == LayerType.DENSE:
                neurons: int = layer.size
                activation: ActivationFunction = layer.activation
                if first_layer is True:
                    model.add(Dense(neurons, activation=activation.value, input_shape=self.config.INPUT_SHAPE))
                    first_layer = False
                else:
                    model.add(Dense(neurons, activation=activation.value))
            elif layer.layer_type == LayerType.CONV2D:
                # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
                filters: int = layer.filters
                kernel_size: Tuple[int, int] = layer.kernel_size
                strides: Tuple[int, int] = layer.strides
                padding: Conv2DPadding = layer.padding
                activation: ActivationFunction = layer.activation
                logging.debug(
                    "filters=%i, kernel_size=%s, strides=%s, padding=%s, activation=%s"
                    % (filters, kernel_size, strides, padding.value, activation.value)
                )
                if first_layer is True:
                    first_layer = False
                    model.add(
                        Conv2D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding.value,
                            activation=activation.value,
                            input_shape=self.config.INPUT_SHAPE,
                        )
                    )
                else:
                    model.add(
                        Conv2D(
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding.value,
                            activation=activation.value,
                        )
                    )
            else:
                raise

        # add final output layers..
        # https://www.machinecurve.com/index.php/2020/03/30/how-to-use-conv2d-with-keras/
        # https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
        model.add(Flatten())
        model.add(Dense(self.config.NB_CLASSES, activation="softmax"))
        model.compile(loss=sparse_categorical_crossentropy, optimizer=self.__optimizer.value, metrics=["accuracy"])

        # return model
        self.model = model
        print("model has been compiled")

    def get_layer(self, layer_index: int) -> Union[DenseLayer, DropoutLayer, Conv2DLayer]:
        return self.__layers[layer_index]

    def get_layers(self) -> List[Union[DenseLayer, DropoutLayer, Conv2DLayer]]:
        return self.__layers

    def get_optimizer(self) -> Optimizers:
        return self.__optimizer

    def get_input_shape(self) -> int:
        return self.config.INPUT_SHAPE

    def get_output_size(self) -> int:
        return self.config.NB_CLASSES

    def decompile(self) -> Any:
        definition = {
            "input_shape": self.config.INPUT_SHAPE,
            "output_size": self.config.NB_CLASSES,
            "optimizer": self.__optimizer.value,
            "layers": [],
        }

        for i in range(0, len(self.__layers)):
            layer = self.decompile_layer(self.__layers[i])
            definition["layers"].append(layer)

        return definition
