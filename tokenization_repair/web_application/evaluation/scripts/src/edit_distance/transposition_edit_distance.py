from typing import Union, List, Tuple, Optional
from functools import lru_cache

import numpy as np

from enum import Enum


@lru_cache()
def edit_distance_with_transpositions(a: Union[str, List],
                                      b: Union[str, List]):
    len_a = len(a) + 1
    len_b = len(b) + 1
    d = np.zeros(shape=(len_a, len_b), dtype=int)
    for i in range(len_a):
        d[i, 0] = i
    for j in range(len_b):
        d[0, j] = j
    for i in range(1, len_a):
        for j in range(1, len_b):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1,
                          d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i, j] = min(d[i, j],
                              d[i - 2, j - 2] + cost)
    return d[len_a - 1, len_b - 1]


class OperationType(Enum):
    NONE = 0
    INSERTION = 1
    DELETION = 2
    REPLACEMENT = 3
    TRANSPOSITION = 4
    UNDEFINED = 99

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.value < other.value


BACKTRACE_OFFSETS = {
    OperationType.INSERTION: np.array([0, -1]),
    OperationType.DELETION: np.array([-1, 0]),
    OperationType.REPLACEMENT: np.array([-1, -1]),
    OperationType.TRANSPOSITION: np.array([-2, -2]),
    OperationType.NONE: np.array([-1, -1])
}


class EditOperation:
    def __init__(self,
                 type: OperationType,
                 position: int,
                 character: Optional[str]):
        self.type = type
        self.position = position
        self.character = character

    def __str__(self):
        return "EditOperation(%s, %i, '%s')" % (str(self.type), self.position, self.character)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return other is not None and \
               self.type == other.type and \
               self.position == other.position and \
               self.character == other.character


def empty_operation_matrix(height: int,
                           width: int) -> List[List[OperationType]]:
    matrix = []
    for i in range(height):
        matrix.append([])
        for j in range(width):
            matrix[-1].append(OperationType.UNDEFINED)
    return matrix


def edit_operation_matrix(a: str,
                          b: str,
                          space_replace: bool) -> Tuple[int, List[List[OperationType]]]:
    len_a = len(a) + 1
    len_b = len(b) + 1
    d = np.zeros(shape=(len_a, len_b), dtype=int)
    operation_matrix = empty_operation_matrix(len_a, len_b)
    for i in range(len_a):
        d[i, 0] = i
        operation_matrix[i][0] = OperationType.DELETION
    for j in range(len_b):
        d[0, j] = j
        operation_matrix[0][j] = OperationType.INSERTION
    for i in range(1, len_a):
        for j in range(1, len_b):
            operation_costs = [
                (d[i, j - 1] + 1, OperationType.INSERTION),
                (d[i - 1, j] + 1, OperationType.DELETION)
            ]
            if a[i - 1] == b[j - 1]:
                operation_costs.append((d[i - 1, j - 1], OperationType.NONE))
            else:
                if space_replace or (a[i - 1] != ' ' and b[j - 1] != ' '):
                        operation_costs.append((d[i - 1, j - 1] + 1, OperationType.REPLACEMENT))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                if space_replace or (a[i - 1] != ' ' and a[i - 2] != ' '):
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    operation_costs.append((d[i - 2, j - 2] + cost, OperationType.TRANSPOSITION))
            ed, operation = min(operation_costs)
            d[i, j] = ed
            operation_matrix[i][j] = operation
    return int(d[len_a - 1, len_b - 1]), operation_matrix


def backtrace(a: str,
              b: str,
              operation_matrix: List[List[OperationType]]) -> List[EditOperation]:
    len_a = len(operation_matrix)
    len_b = len(operation_matrix[0])
    pos = np.array([len_a - 1, len_b - 1])
    edit_operations = []
    while not np.all(pos == [0, 0]):
        edit_type = operation_matrix[pos[0]][pos[1]]
        if edit_type != OperationType.NONE:
            if edit_type == OperationType.INSERTION:
                edit_pos = pos[0]
                character = b[pos[1] - 1]
            elif edit_type == OperationType.TRANSPOSITION:
                edit_pos = pos[0] - 2
                character = None
            elif edit_type == OperationType.REPLACEMENT:
                edit_pos = pos[0] - 1
                character = b[pos[1] - 1]
            elif edit_type == OperationType.DELETION:
                edit_pos = pos[0] - 1
                character = a[pos[0] - 1]
            edit_operations.append(EditOperation(edit_type, edit_pos, character))
        pos += BACKTRACE_OFFSETS[edit_type]
    return edit_operations[::-1]


def edit_operations(a: str,
                    b: str,
                    space_replace: bool) -> List[EditOperation]:
    ed, operation_matrix = edit_operation_matrix(a, b, space_replace)
    operations = backtrace(a, b, operation_matrix)
    return operations
