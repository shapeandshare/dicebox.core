from typing import List, Union

from tensorflow.python.keras.models import Sequential

from ..models.layer import DropoutLayer, DenseLayer
from ..models.optimizers import Optimizers


class NetworkConfig:
    def __init__(self, input_shape: int, output_size: int, optimizer: Optimizers, layers: List[Union[DropoutLayer, DenseLayer]] = None):
        self.input_shape: int = input_shape
        self.output_size: int = output_size
        self.optimizer: Optimizers = optimizer
        if layers is not None:
            self.layers: List[Union[DropoutLayer, DenseLayer]] = layers
        else:
            self.layers: List[Union[DropoutLayer, DenseLayer]] = []
        self.model: Union[Sequential, None] = None
