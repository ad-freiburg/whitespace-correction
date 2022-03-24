import random
import string
import numpy as np


def flip_coin(p):
    return random.uniform(0, 1) < p


def random_character():
    return random.choice(string.ascii_letters)


def random_characters(n):
    return ''.join([random_character() for _ in range(n)])


def random_integers(n, max):
    return [int(x) for x in np.random.uniform(0, max + 1, n)]


class SpellingCorruptor:
    def __init__(self, p):
        self.p = p

    def corrupt(self, sequence):
        n = len(sequence)
        corrupt_sequence = ""
        n_insertions = np.random.binomial(n, self.p / 2)
        insert_chars = random_characters(n_insertions)
        insert_positions = sorted(random_integers(n_insertions, n))
        insert_pos_pointer = 0
        n_deletions = 0
        for i in range(n):
            # insert chars
            while insert_pos_pointer < len(insert_positions) and insert_positions[insert_pos_pointer] == i:
                corrupt_sequence += insert_chars[insert_pos_pointer]
                insert_pos_pointer += 1
            # delete char
            if not flip_coin(self.p / 2):
                corrupt_sequence += sequence[i]
            else:
                n_deletions += 1
        # insert remaining chars
        corrupt_sequence += insert_chars[insert_pos_pointer:]
        return corrupt_sequence


if __name__ == "__main__":
    corruptor = SpellingCorruptor(0.1)
    while True:
        sequence = input("> ")
        if sequence == "exit":
            break
        corrupt = corruptor.corrupt(sequence)
        print(corrupt)
