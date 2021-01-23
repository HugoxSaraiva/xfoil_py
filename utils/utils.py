import ntpath
import uuid


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
