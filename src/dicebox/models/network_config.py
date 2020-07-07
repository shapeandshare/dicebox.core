from typing import List, Union

from tensorflow.python.keras.models import Sequential

from src.dicebox.models.layer import DropoutLayer, DenseLayer
from src.dicebox.models.optimizers import Optimizers


class NetworkConfig:
    def __init__(self, input_shape: int, output_size: int, optimizer: Optimizers, layers: List[Union[DropoutLayer, DenseLayer]]):
        self.input_shape: int = input_shape
        self.output_size: int = output_size
        self.optimizer: Optimizers = optimizer
        self.layers: List[Union[DropoutLayer, DenseLayer]] = layers
        self.model: Union[Sequential, None] = None
