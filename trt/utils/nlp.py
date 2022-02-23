import string
from typing import List, Tuple

import numpy as np


def clean_sequence(sequence: str) -> str:
    """

    Replace all multiple whitespaces, tabs, linebreaks etc. with single whitespaces.

    :param sequence: string
    :return: cleaned string
    """
    return " ".join(sequence.strip().split())


_INCLUDE_ALL = tuple(i for i in range(4))
_EDIT_CHARS = tuple(string.ascii_letters)


def edit_token(token: str,
               rand: np.random.RandomState,
               include: Tuple[int] = _INCLUDE_ALL,
               edit_chars: Tuple[str] = _EDIT_CHARS) -> Tuple[str, List[int]]:
    """

    Perform a random edit operation from {insert, delete, swap, replace} with the token.

    :param token: token string
    :param rand: random state
    :param include: list of integers that represent edit operations from which should be chosen
    :param edit_chars: list of strings to choose from for inserting and replacing
    :return: token with one random edit, list of ints indicating the edits for the positions in the token
    """

    # edit methods: 0 -> insert, 1 -> delete, 2 -> swap, 3 -> replace
    edit_method = rand.choice(include)

    if edit_method == 0:
        char_idx = rand.randint(len(edit_chars))
        token_idx = rand.randint(len(token) + 1)
        _edit_char = edit_chars[char_idx]
        token = token[:token_idx] + _edit_char + token[token_idx:]
        edits = [-1 for _ in range(len(token))]
        for i in range(len(_edit_char)):
            edits[token_idx + i] = 0

    elif edit_method == 1 and len(token) > 1:
        token_idx = rand.randint(len(token))
        token = token[:token_idx] + token[token_idx + 1:]
        edits = [-1 for _ in range(len(token))]
        edits[max(token_idx - 1, 0)] = 1
        edits[min(token_idx, len(token) - 1)] = 1

    elif edit_method == 2 and len(token) > 1:
        token_idx = rand.randint(len(token) - 1)
        token = token[:token_idx] + token[token_idx + 1] + token[token_idx] + token[token_idx + 2:]
        edits = [-1 for _ in range(len(token))]
        edits[token_idx] = 2
        edits[token_idx + 1] = 2

    else:
        token_idx = rand.randint(len(token))
        new_char = token[token_idx]
        while new_char == token[token_idx]:
            char_idx = rand.randint(len(edit_chars))
            new_char = edit_chars[char_idx]
        token = token[:token_idx] + new_char + token[token_idx + 1:]
        edits = [-1 for _ in range(len(token))]
        for i in range(len(new_char)):
            edits[token_idx + i] = 3

    return token, edits
