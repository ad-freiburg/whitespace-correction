import random
import re
import itertools
from typing import List, Union, Tuple, Optional, Callable

import einops
import torch
from torch.nn.utils import rnn


def remove_whitespace(sequence: str) -> str:
    return re.sub(r"\s", "", sequence)


def get_whitespace_operations(from_sequence: str, to_sequence: str) -> List[int]:
    """

    Get the repair sequence that turns from_sequence into to_sequence (after applying the repair_whitespace function)

    :param from_sequence: sequence that the returned repair tokens should be applied to to get the to_sequence
    :param to_sequence: sequence that should result from applying the whitespace operations to the from_sequence
    :return: list of repair tokens
    """
    assert from_sequence.replace(" ", "") == to_sequence.replace(" ", ""), \
        f"make sure from_sequence and to_sequence only differ in whitespaces:\n{from_sequence}\n{to_sequence}"

    from_sequence_ptr = 0
    to_sequence_ptr = 0

    repair_tokens = []

    while from_sequence_ptr < len(from_sequence):  # and to_sequence_ptr < len(to_sequence):
        from_char = from_sequence[from_sequence_ptr]
        to_char = to_sequence[to_sequence_ptr] if to_sequence_ptr < len(to_sequence) else ""

        if from_char == to_char:
            repair_tokens.append(0)
            from_sequence_ptr += 1
            to_sequence_ptr += 1

        elif to_char == " ":
            repair_tokens.append(1)
            from_sequence_ptr += 1
            to_sequence_ptr += 2

        elif from_char == " ":
            repair_tokens.append(2)
            from_sequence_ptr += 1

        else:
            raise ValueError("should not happen")

    assert len(repair_tokens) == len(from_sequence), \
        f"{''.join(str(r) for r in repair_tokens)}\n'{from_sequence}'\n'{to_sequence}'"

    return repair_tokens


def repair_whitespace(sequence: str, repair_tokens: List[int]) -> str:
    """

    Repair the white spacing in the given sequence using the given repair tokens.

    :param sequence: string which has to be repaired
    :param repair_tokens: list with 0's, 1's and 2's indicating to keep the char, insert a whitespace
        or delete a whitespace.
    :return: repaired string
    """
    if len(sequence) > len(repair_tokens):
        repair_tokens.extend([0] * (len(sequence) - len(repair_tokens)))
    else:
        repair_tokens = repair_tokens[:len(sequence)]

    allowed_tokens = {0, 1, 2}
    assert all(token in allowed_tokens for token in repair_tokens), \
        f"only 0's, 1's and 2's are allowed as repair tokens, but got {repair_tokens} for sequence \"{sequence}\""

    sequence_ptr = 0
    token_ptr = 0

    repaired_sequence = ""
    while sequence_ptr < len(sequence):
        char = sequence[sequence_ptr]
        prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else ""
        token = repair_tokens[token_ptr]

        if token == 1 and char != " " and prev_char != " ":
            # if we should insert a whitespace and the current and previous character are not whitespaces,
            # add a whitespace in front of the character
            repaired_sequence += " " + char

        elif token == 2 and char == " ":
            # if we should delete a whitespace and we are at a whitespace, just skip
            pass

        else:
            # keep current character in all other cases
            repaired_sequence += char

        sequence_ptr += 1
        token_ptr += 1

    return repaired_sequence


def clean_sequence(sequence: str) -> str:
    """

    Replace all multiple whitespaces, tabs, linebreaks etc. with single whitespaces.

    :param sequence: string
    :return: cleaned string
    """
    # about 5 times faster than re.sub("\s+", " ", sequence)
    return " ".join(sequence.strip().split())


def find_word_boundaries(s: str) -> List[Tuple[int, int]]:
    # this function assumes that s is cleaned with clean_sequence above
    boundaries = []
    start_idx = 0
    for word in s.split():
        end_idx = start_idx + len(word)
        boundaries.append((start_idx, end_idx))
        start_idx = end_idx + 1
    return boundaries


def random_character_substring(
        s: str,
        num_chars: int,
        rand: random.Random
) -> Tuple[int, int]:
    if s == "":
        return 0, 0
    start = rand.randint(0, max(0, len(s) - num_chars))
    return start, min(len(s), start + num_chars)


def _find_subsequences_with_sum_close_to_but_max_k(
        values: List[int],
        k: int
) -> List[Tuple[int, int]]:
    if len(values) == 0:
        return []
    # this is linear
    cum_values = list(itertools.accumulate(values))
    if cum_values[-1] <= k:
        return [(0, len(cum_values))]
    start = 0
    # move start pointer to first valid start position (element smaller or equal to k)
    while start < len(values) and values[start] > k:
        start += 1
    if start >= len(values):
        return []
    end = start
    subsequences = []
    while start < len(cum_values) and end < len(cum_values):
        next_end_v = values[end + 1] if end + 1 < len(cum_values) else 0
        if next_end_v > k:
            subsequences.append((start, end + 1))
            start = end + 2
            end = start
        else:
            cum_next_end_v = cum_values[end] + next_end_v
            cum_up_to_start = cum_values[start] - values[start]
            if cum_next_end_v - cum_up_to_start > k:
                if len(subsequences) == 0 or subsequences[-1][1] < end + 1:
                    subsequences.append((start, end + 1))
                start += 1
            else:
                end += 1
    if start != end:
        subsequences.append((start, end))
    return subsequences


def random_byte_substring(
        s: str,
        max_bytes: int,
        rand: random.Random
) -> Tuple[int, int]:
    if s == "":
        return 0, 0
    num_bytes = list(len(c.encode("utf8")) for c in s)
    possible_subsequences = _find_subsequences_with_sum_close_to_but_max_k(num_bytes, max_bytes)
    return rand.choice(possible_subsequences)


def find_substring_ignoring_spaces(s: str, substring: str) -> Tuple[int, int]:
    substring_pattern = re.compile(r"\s?".join(
        re.escape(char) for char in substring if char != " "
    ))
    match = substring_pattern.search(s)
    assert match is not None, f"could not find substring \"{substring}\" in \"{s}\""
    return match.start(), match.end()
