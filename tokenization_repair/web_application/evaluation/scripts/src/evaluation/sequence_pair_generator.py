import random

from src.helper.files import read_sequences


class SequencePairGenerator:
    def __init__(self, correct_path, corrupt_path):
        self.correct_path = correct_path
        self.corrupt_path = corrupt_path

    def sequence_pairs(self, n=None, seed=None):
        correct_sequences = list(read_sequences(self.correct_path))
        corrupt_sequences = list(read_sequences(self.corrupt_path))
        pairs = list(zip(correct_sequences, corrupt_sequences))
        if seed is not None:
            rdm = random.Random(seed)
            rdm.shuffle(pairs)
        if n is not None:
            pairs = pairs[:n]
        return pairs
