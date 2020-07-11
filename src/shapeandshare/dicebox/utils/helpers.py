import errno
import math
import os
import struct


def lucky(noise=0.0) -> bool:
    """Simple function to determine whether or not a thing occurs.

    :param noise: floating point value for noise
    :return: boolean
    """
    if noise == 1.0:
        return True
    if noise == 0.0:
        return False
    if noise > float(ord(struct.unpack('c', os.urandom(1))[0])) / 255:
        return True
    return False


def random_index(set_size: int) -> int:
    rand: float = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
    return int(math.ceil(rand * set_size))


def random_index_between(min_index: int = 0, max_index: int = 1) -> int:
    if min_index > max_index:
        raise Exception('max must be greater than or equal to the min')

    rand: float = float(ord(struct.unpack('c', os.urandom(1))[0])) / 255
    delta: int = max_index - min_index
    delta_offset: int = int(math.ceil(rand * delta))
    return min_index + delta_offset


def dicebox_random() -> float:
    return float(ord(struct.unpack('c', os.urandom(1))[0])) / 255


# A dropout layer can is [0, 1), we can not actually use a '1' value.
def random_strict() -> float:
    new_random_number: float = dicebox_random()
    while new_random_number >= 1.0:
        new_random_number = dicebox_random()
    return new_random_number


###############################################################################
# Allows for easy directory structure creation
# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
###############################################################################
def make_sure_path_exists(path) -> None:
    try:
        if os.path.exists(path) is False:
            os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise exception

