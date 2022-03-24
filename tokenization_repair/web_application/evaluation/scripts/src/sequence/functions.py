from typing import Tuple, List, Optional


def unify_spaces(sequence: str) -> str:
    """
    Replace all kinds of spaces by normal spaces.
    """
    snippets = sequence.split()
    unified = ' '.join(snippets)
    return unified


def map_positions(sequence1: str,
                  sequence2: str) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """
    Maps the changed sequence to the original sequence.

    :param sequence1: A sequence.
    :param sequence2: Another sequence that differs from sequence1 only in spaces.
    :return: Two lists where one maps from the positions in sequence1 to sequence2,
        and the other the other way around. Inserted positions are mapped to None.
    """
    fwd_mapping = []  # seq1 -> seq2
    bwd_mapping = []  # seq2 -> seq1
    i1 = 0
    i2 = 0
    while i1 < len(sequence1) or i2 < len(sequence2):
        char1 = sequence1[i1] if i1 < len(sequence1) else None
        char2 = sequence2[i2] if i2 < len(sequence2) else None
        if char1 == char2:
            fwd_mapping.append(i2)
            bwd_mapping.append(i1)
            i1 += 1
            i2 += 1
        elif char1 == ' ':
            fwd_mapping.append(None)
            i1 += 1
        elif char2 == ' ':
            bwd_mapping.append(None)
            i2 += 1
        else:
            raise Exception("Sequences differ in characters unequal to spaces.")
    return fwd_mapping, bwd_mapping


def get_space_positions_in_merged(sequence: str):
    space_positions = [i for i, char in enumerate(sequence) if char == ' ']
    space_positions = [pos - i for i, pos in enumerate(space_positions)]
    return set(space_positions)


def remove_spaces(sequence: str) -> str:
    return sequence.replace(' ', '')