import numpy as np


def empty_matrix(batch_size, dimensions):
    return np.array([]).reshape((batch_size, 0, dimensions))
