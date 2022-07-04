import re
from enum import Enum
from typing import List, Union


class WhitespaceCorrectionTokens(Enum):
    KEEP_CHAR = "#"  # "<k>"
    INSERT_WS = "_"  # "<iw>"
    DELETE_WS = "x"  # "<dw>"


def get_correction_tokens(repair_sequence: str) -> List[str]:
    return re.findall(f"({WhitespaceCorrectionTokens.DELETE_WS.value}"
                      f"|{WhitespaceCorrectionTokens.INSERT_WS.value}"
                      f"|{WhitespaceCorrectionTokens.KEEP_CHAR.value})", repair_sequence)


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


def repair_whitespace(sequence: str, repair_sequence: Union[str, List[int]]) -> str:
    """

    Repair the white spacing in the given sequence using the given repair_sequence.

    :param sequence: string which has to be repaired
    :param repair_sequence: string containing instructions on how to repair or list with 0's, 1's and 2's
        indicating to keep the char, insert a whitespace or delete a whitespace.
        indicate padding positions.
    :return: repaired string
    """
    if isinstance(repair_sequence, str):
        repair_tokens = []
        for c in repair_sequence:
            if c == WhitespaceCorrectionTokens.KEEP_CHAR.value:
                repair_tokens.append(0)
            elif c == WhitespaceCorrectionTokens.INSERT_WS.value:
                repair_tokens.append(1)
            elif c == WhitespaceCorrectionTokens.DELETE_WS.value:
                repair_tokens.append(2)
            else:
                raise ValueError(f"Only #|x|_ are allowed in repair sequence, but got {c}")
    else:
        repair_tokens = repair_sequence

    if len(sequence) > len(repair_tokens):
        repair_tokens.extend([0] * (len(sequence) - len(repair_tokens)))
    else:
        repair_tokens = repair_tokens[:len(sequence)]

    allowed_tokens = {0, 1, 2}
    assert all([token in allowed_tokens for token in repair_tokens]), \
        f"Only 0's, 1's and 2's are allowed as repair tokens, but got {repair_tokens} for sequence \"{sequence}\""

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
