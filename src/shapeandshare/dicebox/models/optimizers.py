from enum import Enum
from random import choices


class Optimizers(Enum):
    ADADELTA = "adadelta"
    ADAGRAD = "adagrad"
    ADAM = "adam"
    ADAMAX = "adamax"
    FTRL = "ftrl"
    NADAM = "nadam"
    RMSPROP = "rmsprop"
    SGD = "sgd"


def select_random_optimizer() -> Optimizers:
    return choices(
        [
            Optimizers.ADADELTA,
            Optimizers.ADAGRAD,
            Optimizers.ADAM,
            Optimizers.ADAMAX,
            Optimizers.FTRL,
            Optimizers.NADAM,
            Optimizers.RMSPROP,
            Optimizers.SGD,
        ]
    )[0]
