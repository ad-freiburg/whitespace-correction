import numpy as np


def tp(ground_truth_set, prediction_set):
    return ground_truth_set.intersection(prediction_set)


def fp(ground_truth_set, prediction_set):
    return prediction_set.difference(ground_truth_set)


def fn(ground_truth_set, prediction_set):
    return ground_truth_set.difference(prediction_set)


def precision_recall_f1(n_tp, n_fp, n_fn):
    precision = 0 if n_tp + n_fp == 0 else n_tp / (n_tp + n_fp)
    recall = 0 if n_tp + n_fn == 0 else n_tp / (n_tp + n_fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def ed_fraction_resolved(ed_before, ed_after):
    return (ed_before - ed_after) / ed_before


def perplexity(probabilities, n=None):
    if n is None:
        n = len(probabilities)
    return np.power(2, -np.sum(np.log2(probabilities)) / n)


def crossentroy(probabilities):
    return -np.mean(np.log(probabilities))
