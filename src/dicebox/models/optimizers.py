from enum import Enum


class Optimizers(Enum):
    RMSPROP = 'rmsprop'
    ADAM = 'adam'
    SGD = 'sgd'
    ADAGRAD = 'adagrad'
    ADADELTA = 'adadelta'
    ADAMAX = 'adamax'
    NADAM = 'nadam'
