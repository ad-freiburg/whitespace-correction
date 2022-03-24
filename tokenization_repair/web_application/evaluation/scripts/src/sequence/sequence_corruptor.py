import numpy.random
from enum import Enum
import string
#import global_values


"""def get_alphabetical_char_dict():
    symbols = [' '] + list(string.ascii_letters) \
              + [global_values.EOS_SYMBOL, global_values.SOS_SYMBOL, global_values.UNKNOWN_SYMBOL]
    dictionary = {i: symbols[i] for i in range(len(symbols))}
    return dictionary"""


class CorruptionType(Enum):
    INSERTION = 0
    DELETION = 1


class Corruption:
    def __init__(self, type, position, character):
        self.type = type
        self.position = position
        self.character = character

    def __str__(self):
        return str((self.type, self.position, self.character))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.type == other.type and self.position == other.position and self.character == other.character

    def __hash__(self):
        return (self.type, self.position, self.character).__hash__()

    def __lt__(self, other):
        return False


class SequenceCorruptor:
    def __init__(self, tokenization=True, n=None, p=None, insert=True, delete=True, seed=None, char_dict=None):
        self.tokenization = tokenization
        self.n = n
        self.p = p
        self.insert = insert
        self.delete = delete
        self.rdm = numpy.random.RandomState(seed)
        self.char_dict = char_dict

    def corrupt(self, sequence):
        if self.tokenization:
            return self.corrupt_tokens(sequence)
        else:
            return self.corrupt_characters(sequence)

    def _pick_from_possible_corruptions(self, possible_corruptions):
        if self.n is not None:
            sample_size = min(len(possible_corruptions), self.n)
            choice = self.rdm.choice(len(possible_corruptions), size=sample_size, replace=False)
            selected_corruptions = [possible_corruptions[i] for i in choice]
        elif self.p is not None:
            sample = self.rdm.uniform(0, 1, len(possible_corruptions))
            selected_corruptions = [possible_corruptions[i] for i in range(len(possible_corruptions))
                                    if sample[i] < self.p]
        else:
            raise NotImplementedError("Either n or p must not be None.")
        return set(selected_corruptions)

    def corrupt_tokens(self, sequence):
        tokens = sequence.split(' ')
        possible_corruptions = []
        if self.delete:
            #space_positions = [i for i in range(len(sequence)) if sequence[i] == ' ']
            possible_deletions = [(i, CorruptionType.DELETION) for i in range(1, len(tokens))]
            possible_corruptions += possible_deletions
        if self.insert:
            possible_insertions = [(i, CorruptionType.INSERTION) for i in range(0, len(tokens)) if len(tokens[i]) > 1]
            possible_corruptions += possible_insertions
        selected_corruptions = self._pick_from_possible_corruptions(possible_corruptions)
        corrupted = ""
        corruptions = []
        for i, token in enumerate(tokens):
            # deletion:
            if i > 0 and (i, CorruptionType.DELETION) not in selected_corruptions:
                corrupted += ' '
            elif (i, CorruptionType.DELETION) in selected_corruptions:
                corruptions.append(Corruption(CorruptionType.DELETION, len(corrupted), ' '))
            # insertion:
            if (i, CorruptionType.INSERTION) in selected_corruptions:
                split_pos = int(self.rdm.uniform(1, len(token)))
                appendix = token[:split_pos] + ' ' + token[split_pos:]
                corruptions.append(Corruption(CorruptionType.INSERTION, len(corrupted) + split_pos, ' '))
            else:
                appendix = token
            corrupted += appendix
        return corruptions, corrupted

    def _random_character(self):
        return self.char_dict[int(self.rdm.uniform(0, len(self.char_dict) - 3))]

    def corrupt_characters(self, sequence):
        possible_corruptions = []
        if self.delete:
            possible_deletions = [(i, CorruptionType.DELETION) for i in range(len(sequence))]
            possible_corruptions += possible_deletions
        if self.insert:
            possible_insertions = [(i, CorruptionType.INSERTION) for i in range(len(sequence) + 1)]
            possible_corruptions += possible_insertions
        selected_corruptions = self._pick_from_possible_corruptions(possible_corruptions)
        corrupted = ""
        corruptions = []
        for i in range(len(sequence) + 1):
            # deletion:
            if (i, CorruptionType.DELETION) in selected_corruptions:
                corruptions.append(Corruption(CorruptionType.DELETION, len(corrupted), sequence[i]))
            elif i < len(sequence):
                corrupted += sequence[i]
            # insertion:
            if (i, CorruptionType.INSERTION) in selected_corruptions:
                char = self._random_character()
                corruptions.append(Corruption(CorruptionType.INSERTION, len(corrupted), char))
                corrupted += char
        return corruptions, corrupted


if __name__ == "__main__":
    from wiki_dataset import get_dictionaries

    _, ix2char = get_dictionaries()
    sc = SequenceCorruptor(n=1, insert=False, tokenization=False, char_dict=ix2char)
    s = "Why is University so much fun?"
    print(s)
    corruptions, corrupted = sc.corrupt(s)
    print([str(c) for c in corruptions])
    print(corrupted)
