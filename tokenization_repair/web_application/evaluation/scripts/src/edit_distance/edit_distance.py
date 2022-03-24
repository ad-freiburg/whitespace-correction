import numpy as np
from enum import Enum


class OperationTypes(Enum):
    INSERT = 0
    DELETE = 1
    REPLACE = 2


def char_equals(char1, char2):
    return char1 == char2 or (char1 in [' ', '\xa0'] and char2 in [' ', '\xa0'])


def levenshtein(s, t, limit=None, substitutions=True, return_matrix=False):
    m = len(s)
    n = len(t)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = range(n + 1)
    d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if char_equals(t[i - 1], s[j - 1]):
                d[i, j] = d[i - 1, j - 1]
            else:
                if substitutions:
                    d[i, j] = min(d[i - 1, j] + 1,
                                  d[i, j - 1] + 1,
                                  d[i - 1, j - 1] + 1)
                else:
                    d[i, j] = min(d[i - 1, j] + 1,
                                  d[i, j - 1] + 1)
    if return_matrix:
        return d[-1, -1], d
    return d[-1, -1]


def limited_levenshtein(s, t, limit=None, return_matrix=False, substitutions=True):
    if limit is None:
        return levenshtein(s, t, return_matrix=return_matrix, substitutions=substitutions)

    m = len(s)
    n = len(t)

    if abs(m - n) > limit and not return_matrix:
        return limit + 1

    d = np.full((n + 1, m + 1), -1, dtype=int)
    d[:, 0] = range(n + 1)
    d[0, :] = range(m + 1)

    for i in range(1, n + 1):
        left = max(1, i - limit)
        right = min(m + 1, i + limit + 1)
        for j in range(left, right):
            if char_equals(t[i - 1], s[j - 1]):
                d[i, j] = d[i - 1, j - 1]
            else:
                insertion_value = (limit + 1) if d[i - 1, j] == -1 else (d[i - 1, j] + 1)
                deletion_value = (limit + 1) if d[i, j - 1] == -1 else (d[i, j - 1] + 1)
                if substitutions:
                    substitution_value = d[i - 1, j - 1] + 1
                    d[i, j] = min(insertion_value,
                                  deletion_value,
                                  substitution_value)
                else:
                    d[i, j] = min(insertion_value,
                                  deletion_value)
    distance = (limit + 1) if d[-1, -1] == -1 else d[-1, -1]
    if return_matrix:
        return distance, d
    return distance


def get_operations(s, t, edit_distance_matrix, substitutions=True):
    y = edit_distance_matrix.shape[0] - 1
    x = edit_distance_matrix.shape[1] - 1
    operations = []
    while y > 0 or x > 0:
        if x > 0 and y > 0 and char_equals(s[x - 1], t[y - 1]) and edit_distance_matrix[y - 1, x - 1] != -1:
            x = x - 1
            y = y - 1
        else:
            # insert:
            if y == 0 or edit_distance_matrix[y - 1, x] == -1:
                insert_value = float("inf")
            else:
                insert_value = edit_distance_matrix[y - 1, x]
            # delete:
            if x == 0 or edit_distance_matrix[y, x - 1] == -1:
                delete_value = float("inf")
            else:
                delete_value = edit_distance_matrix[y, x - 1]
            neighbors = [insert_value, delete_value]
            if substitutions:
                if x > 0 and y > 0 and edit_distance_matrix[y - 1, x - 1] != -1:
                    neighbors.append(edit_distance_matrix[y - 1, x - 1])
            best_operation = np.argmin(neighbors)
            if best_operation == 0:
                y = y - 1
                operations.append((OperationTypes.INSERT, x, t[y]))
            elif best_operation == 1:
                x = x - 1
                operations.append((OperationTypes.DELETE, x, s[x]))
            else:
                y = y - 1
                x = x - 1
                operations.append((OperationTypes.REPLACE, x, s[x], t[y]))
    operations = operations[::-1]
    return operations


def get_edit_operations(source, target, substitutions):
    d, matrix = levenshtein(source, target, return_matrix=True, substitutions=substitutions)
    operations = get_operations(source, target, matrix, substitutions=substitutions)
    return operations


if __name__ == "__main__":
    """print(levenshtein("kitten", "sitting"))
    print(levenshtein("kitten", "sitting", substitutions=False))
    print(levenshtein("Saturday", "Sunday"))
    print(levenshtein("Saturday", "Sunday", substitutions=False))
    print(levenshtein("xxxxxxx", "xxxxx", return_matrix=True)[1])
    print()
    print(limited_levenshtein("kitten", "kitchen", limit=2, return_matrix=True, substitutions=False)[1])"""

    # operations test:
    substitutions = False
    s = "whose"
    t = "goose"

    s = """Well fleshed, long, deep and symmetrical. Well proportioned, with shoulders strong, smooth and blending well into body, well placed, fitting smoothly upon chest, which should be deep and wide; forearm well muscled; long, broad, straight level back; well sprung ribs; thick, wide and long loins well covered with firm flesh; hips wide and smooth. Rump long, hind quarters well developed, long and wide with dock well set on and twist deep and full, legs of mutton full, deep and well-muscled."""
    #t = """Wellh fleshed, ong, deep ’and symmetri<cal. Well proportioned, withshoulders( st1rong, s(moth and Zlendi ng welO intUo ody, wellplaced, fitting smoothly uponK chest, whic skhoul%d beep and wide; forearm well muschyed; long, b>road, st raght level ack;weekll sprung ribs; thick, wide ad long loin,s wTel cerkd with firm flesh; hips widoe and smooth. Rumplong, hind quarteLrs  well developed, long and widz with dock ell et on and twist deep and full, legs of mutYton full, deep and wesll-musce.ü"""
    t = """Well fleshed, on, deep ’and symmetrical. Well proportioned, with shoulders( strong, moth and Glen di ng well into body, well placed, fitting smoothly upon chest, which should beep and wide; forearm well muschyed; long, broad, straight level ack;weekll sprung ribs; thick, wide ad long lo ins Tel clerk with firm flesh; hips wide and smooth. Rump long, hind quarters  well developed, long and wide with dock ell et on and twist deep and full, legs of mutton full, deep and wesll-musce.ü"""
    d, matrix = levenshtein(s, t, return_matrix=True, substitutions=substitutions)
    print(d)
    print(matrix)
    operations = get_operations(s, t, matrix, substitutions=substitutions)
    print(operations)
