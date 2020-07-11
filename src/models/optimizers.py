from enum import Enum
from random import random

class Optimizers(Enum):
    RMSPROP = 'rmsprop'
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    ADADELTA = 'adadelta'
    ADAMAX = 'adamax'
    NADAM = 'nadam'

    @staticmethod
    def select_random_optimizer():
        return random.choice(
            Optimizers.RMSPROP,
            Optimizers.ADAM,
            Optimizers.SGD,
            Optimizers.ADAGRAD,
            Optimizers.ADADELTA,
            Optimizers.ADAMAX,
            Optimizers.NADAM
        )
