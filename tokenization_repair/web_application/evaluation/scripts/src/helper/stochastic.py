import random
import numpy as np


def flip_coin(random_object: random.Random,
              p: float) -> bool:
    value = random_object.uniform(0, 1)
    return value < p


def log_likelihood(probs):
    return np.sum(np.log(probs))
