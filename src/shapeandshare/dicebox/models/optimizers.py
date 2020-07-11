from enum import Enum
from random import choices


class Optimizers(Enum):
    RMSPROP = 'rmsprop'
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    ADADELTA = 'adadelta'
    ADAMAX = 'adamax'
    NADAM = 'nadam'


def select_random_optimizer() -> Optimizers:
    return choices([
        Optimizers.RMSPROP,
        Optimizers.ADAM,
        Optimizers.SGD,
        Optimizers.ADAGRAD,
        Optimizers.ADADELTA,
        Optimizers.ADAMAX,
        Optimizers.NADAM
    ])[0]
