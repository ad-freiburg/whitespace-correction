import re
import string
from typing import List, Tuple

import ftfy

import numpy as np


def clean_sequence(sequence: str) -> str:
    """

    Replace all multiple whitespaces, tabs, linebreaks etc. with single whitespaces.

    :param sequence: string
    :return: cleaned string
    """
    return " ".join(sequence.strip().split())


def clean_text(sequence: str) -> str:
    """

    Fixes quotes and unicode issues using ftfy.

    :param sequence: string
    :return: cleaned string
    """
    return ftfy.fix_text(sequence)


def is_valid_sequence(sequence: str, min_length: int = 10) -> bool:
    """

    Check if a string is a valid sequence in the sense that it is a proper sentence/expression.

    :param sequence: string
    :param min_length: minimum length of string
    :return: bool whether string is valid
    """
    # from tokenization repair repo
    f = re.compile(r" [.,;]( |$)|<|>|\"\"|\(\)| ' |\([,;]|colspan")
    if f.search(sequence) is not None:
        return False
    # if sequence is smaller than min_length characters its invalid
    if len(sequence) < min_length:
        return False
    # sequence must contain at least one standard character to be valid
    contains_chars = re.compile(r"[a-zA-Z]+")
    if contains_chars.search(sequence) is None:
        return False
    # check if sequence contains some xml/html markup
    contains_markup = re.compile(r"<.*?>|</.*?>")
    if contains_markup.search(sequence) is not None:
        return False
    # if sequence passes all the tests its valid
    return True


_APOSTROPH = {r"\s(['`]\w+)": r"\1",
              r"\s(n['`]t\s)": r"\1"}

_PUNCTUATION = {r"\s([.,?!;:])": r"\1"}


def tokens_to_text(tokens: List[str]) -> str:
    """

    Bring tokens together to a proper string. Just joining with whitespaces is not enough,
    e.g. ["I", "have", "n't"] should get "I haven't" and not "I have n't".

    :param tokens: list of tokens
    :return: proper string
    """
    tokens = [token.strip() for token in tokens]

    sequence = " ".join(tokens)
    sequence = clean_text(sequence)

    for pattern, sub in _APOSTROPH.items():
        sequence = re.sub(pattern, sub, sequence)

    for pattern, sub in _PUNCTUATION.items():
        sequence = re.sub(pattern, sub, sequence)

    return clean_sequence(sequence)


_INCLUDE_ALL = [i for i in range(4)]
_EDIT_CHARS = list(string.ascii_letters)


def edit_token(token: str,
               rand: np.random.RandomState,
               include: List[int] = _INCLUDE_ALL,
               edit_chars: List[str] = _EDIT_CHARS) -> Tuple[str, List[int]]:
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
