from typing import Dict, Any, List

import numpy as np
import math
from itertools import count


def sort_dict_by_value(d, reverse=True):
    """
    Sorts a dictionary by its values.
    :param d: dictionary of key-value pairs
    :param reverse: set False to sort in increasing order, else decreasing
    :return: list of sorted key-value pairs
    """
    sorted_keys = sorted(d, key=d.get, reverse=reverse)
    return [(key, d[key]) for key in sorted_keys]


def sum_entries(dictionary):
    return np.sum([dictionary[key] for key in dictionary])


def gather(matrix, indices):
    return np.asarray([matrix[i, index] for i, index in enumerate(indices)])


def argmax_and_max(vector):
    amax = np.argmax(vector)
    return amax, vector[amax]


def top_k_indices(vector, k):
    return np.argpartition(vector, -k)[-k:]


def top_k_indices_sorted(vector, k):
    top_indices = top_k_indices(vector, k)
    top_entries = np.asarray(vector)[top_indices]
    sort_indices = np.argsort(top_entries)[::-1]
    top_indices_sorted = top_indices[sort_indices]
    return top_indices_sorted


def sorted_position(vector, index):
    value = vector[index]
    num_greater = np.sum(np.asarray(vector) > value)
    return num_greater


def izip(*args):
    return zip(count(), *args)


def frequency_rank(frequencies: Dict[Any, int]) -> Dict[Any, int]:
    """
    Gives the rank of the elements in the dictionary when sorting by descending frequency.

    :param frequencies: dictionary mapping something -> frequency of something
    :return: dictionary mapping something -> rank of something
    """
    frequency_key_pairs = [(frequencies[key], key) for key in frequencies]
    frequency_key_pairs = sorted(frequency_key_pairs, reverse=True)
    ranks = {key: i for i, (frequency, key) in enumerate(frequency_key_pairs)}
    return ranks


def revert_dictionary(dict: Dict) -> Dict:
    """
    Gives the inverse mapping of a dictionary.

    :param dict: key -> value
    :return: dictionary value -> key
    """
    return {dict[key]: key for key in dict}


def unique_on_sorted(sorted_list: List) -> List:
    if len(sorted_list) == 0:
        return []
    unique_elements = [sorted_list[0]]
    for elem in sorted_list[1:]:
        if unique_elements[-1] != elem:
            unique_elements.append(elem)
    return unique_elements


def select_most_frequent(dictionary: Dict[Any, int], n: int) -> Dict[Any, int]:
    sorted = sort_dict_by_value(dictionary)
    dictionary = {key: value for key, value in sorted[:n]}
    return dictionary


def deep_copy(dictionary):
    copy = {}
    for key in dictionary:
        if isinstance(dictionary[key], list) or isinstance(dictionary[key], dict):
            copy[key] = dictionary[key].copy()
        else:
            copy[key] = dictionary[key]
    return copy


def insert_into_sorted(array, element):
    n = len(array)
    left = 0
    right = n
    while left != right:
        probe = (left + right) // 2
        comp_elem = array[probe]
        if element == comp_elem:
            left = right = probe
        elif element < comp_elem:
            right = probe
        else:
            left = probe + 1
    return array[:left] + [element] + array[right:]
