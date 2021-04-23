import ntpath
import uuid
import itertools
import json
import numpy
import logging
from functools import wraps


def random_string():
    return str(uuid.uuid4().hex)


def add_prefix_suffix(string: str, prefix="", suffix=""):
    """
    Function that adds suffix to another string if it is not already in the string
    :param string: string
    :param prefix: string
    :param suffix: string
    :return: string with suffix
    """
    new_string = string if string.startswith(prefix) else prefix + string
    new_string = new_string if string.endswith(suffix) else new_string + suffix
    return new_string


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def zip_longest_modified(*args, fillvalue=None):
    """
    Modified itertools.zip_longest function that accepts different fillvalue for each column.
    """
    # zip_longest('ABCD', 'xy', '123', fillvalue=['A', 'w', '0']) --> Ax1 By2 Cw3 Dw0
    iterators = [iter(it) for it in args]
    num_active = len(iterators)
    if not num_active:
        return
    while True:
        values = []
        for i, it in enumerate(iterators):
            try:
                value = next(it)
            except StopIteration:
                num_active -= 1
                if not num_active:
                    return
                iterators[i] = itertools.repeat(fillvalue[len(values)])
                value = fillvalue[len(values)]
            values.append(value)
        yield tuple(values)


def log(message, log_level=logging.INFO):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            logging.log(log_level, message)
            return function(*args, **kwargs)
        return wrapper
    return decorator


class XfoilEncoder(json.JSONEncoder):
    """ Special json encoder for XFoil class, which includes numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {member: getattr(obj, member)
                    for member in dir(obj)
                    if not member.startswith('_') and
                    not hasattr(getattr(obj, member), '__call__')}
        return super().default(obj)