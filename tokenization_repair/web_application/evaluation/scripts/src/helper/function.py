import numpy as np
import time


def prob2score(probability, threshold):
    return (probability - threshold) / (1 - threshold)


def pick_from_probabilities(probs):
    p = np.random.uniform(0, 1)
    i = 0
    cumulative = probs[0]
    while cumulative < p and i < len(probs) - 1:
        i += 1
        cumulative += probs[i]
    return i


def current_time_string():
    return time.strftime("%Y_%m_%d_%H:%M:%S")
