import string
import random
from typing import List, Tuple

_INCLUDE_ALL = tuple(i for i in range(4))
_EDIT_CHARS = tuple(string.ascii_letters)


def edit_token(token: str,
               rand: random.Random,
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
        char_idx = rand.randint(0, len(edit_chars) - 1)
        token_idx = rand.randint(0, len(token))
        _edit_char = edit_chars[char_idx]
        token = token[:token_idx] + _edit_char + token[token_idx:]
        edits = [-1 for _ in range(len(token))]
        for i in range(len(_edit_char)):
            edits[token_idx + i] = 0

    elif edit_method == 1 and len(token) > 1:
        token_idx = rand.randint(0, len(token) - 1)
        token = token[:token_idx] + token[token_idx + 1:]
        edits = [-1 for _ in range(len(token))]
        edits[max(token_idx - 1, 0)] = 1
        edits[min(token_idx, len(token) - 1)] = 1

    elif edit_method == 2 and len(token) > 1:
        token_idx = rand.randint(0, len(token) - 2)
        token = token[:token_idx] + token[token_idx + 1] + token[token_idx] + token[token_idx + 2:]
        edits = [-1 for _ in range(len(token))]
        edits[token_idx] = 2
        edits[token_idx + 1] = 2

    elif edit_method == 3 and len(token) > 1:
        token_idx = rand.randint(0, len(token) - 1)
        new_char = token[token_idx]
        while new_char == token[token_idx]:
            char_idx = rand.randint(0, len(edit_chars) - 1)
            new_char = edit_chars[char_idx]
        token = token[:token_idx] + new_char + token[token_idx + 1:]
        edits = [-1 for _ in range(len(token))]
        for i in range(len(new_char)):
            edits[token_idx + i] = 3

    else:
        edits = []

    return token, edits
