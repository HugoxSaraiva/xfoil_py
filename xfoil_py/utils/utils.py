import ntpath
import uuid
import itertools
import json
import re
import numpy as np
import logging
from functools import wraps


def random_string():
    return str(uuid.uuid4().hex)


def add_prefix_suffix(string: str, prefix="", suffix=""):
    """
    Function that adds suffix to another string if it is not already in the string. Returns None if string is None.
    :param string: string
    :param prefix: string
    :param suffix: string
    :return: string with suffix
    """
    if string is None:
        return None
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


def __read_stdout(self):
    logging.info("Reading stdout")
    exp = "((?<=x[/]c\s=\s{2})\d\.\d*|\d+(?=\s{3}rms)|" \
          "(?<=(?:Cm|CL|CD)\s=\s)[\s-]?\d*\.\d*|" \
          "(?<=a\s=\s)[\s-]?\d*\.\d*|" \
          "(?<=CDp\s=\s)[\s-]?\d*\.\d*)"
    regex = re.compile(exp)

    data = regex.findall(self._stdout)
    a = []
    cl = []
    cd = []
    cdp = []
    cm = []
    xtr_top = []
    xtr_bottom = []

    # Pick data from only the last iteration for each angle
    i = 0
    while i < len(data):
        iteration = float(data[i + 2])
        if i + 10 > len(data) or iteration > float(data[i + 10]):
            xtr_top.append(float(data[i]))
            xtr_bottom.append(float(data[i + 1]))
            a.append(float(data[i + 3]))
            cl.append(float(data[i + 4]))
            cm.append(float(data[i + 5]))
            cd.append(float(data[i + 6]))
            cdp.append(float(data[i + 7]))
        i = i + 8
    return {'a': np.array(a), 'cl': np.array(cl), 'cd': np.array(cd), 'cdp': np.array(cdp), 'cm': np.array(cm),
            'xtr_top': np.array(xtr_top), 'xtr_bottom': np.array(xtr_bottom)}


class XfoilEncoder(json.JSONEncoder):
    """ Special json encoder for XFoil class, which includes numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return {member: getattr(obj, member)
                    for member in dir(obj)
                    if not member.startswith('_') and
                    not hasattr(getattr(obj, member), '__call__')}
        return super().default(obj)
