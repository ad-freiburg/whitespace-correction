from enum import Enum

from src.sequence.corruption import CorruptionType
from src.tokens.token_counter import is_word


def revert_corruptions(sequence, corruptions):
    characters_at_positions = [sequence[i] for i in range(len(sequence))]
    # delete inserted chars
    for corruption in corruptions:
        if corruption.type == CorruptionType.INSERTION:
            characters_at_positions[corruption.position] = ""
    # insert deleted chars
    characters_at_positions = [""] + characters_at_positions
    for corruption in corruptions:
        if corruption.type == CorruptionType.DELETION:
            characters_at_positions[corruption.position] += corruption.character
    repaired = "".join(characters_at_positions)
    return repaired


class SplitStates(Enum):
    SPACE = 0
    LETTER = 1
    NUMERIC = 2
    SPECIAL = 3


def is_space(character):
    return character == ' ' or character == '\xa0'


def is_letter(character):
    return character.isalpha()


def is_uppercase(character):
    return is_letter(character) and character.upper() == character


def is_lowercase(character):
    return is_letter(character) and character.lower() == character


def is_number(character):
    return character.isnumeric()


def is_special_character_exception(char, before, after):
    # 100,000.123
    if char in [',', '.'] and len(before) > 0 and len(after) > 0 and is_number(before[-1]) and is_number(after[0]):
        return True
    if char in ["'", "’"] and len(before) > 0 and len(after) > 0:
        # ...'s / ...'t
        if is_letter(before[-1]) and after[0] in ['s', 't']:
            return True
        # I'm
        if before[-1] == 'I' and after[0] == 'm':
            return True
    # bla-bli
    if char == '-' and len(before) > 0 and len(after) > 0 and is_letter(before[-1]) and is_letter(after[0]):
        return True
    return False


def is_special_character(char):
    if char.isalpha() or char.isnumeric():
        return False
    return True


def pretokenize(sequence):
    tokens = []
    editable = []
    can_append = False
    for i, char in enumerate(sequence):
        before = sequence[i - 1] if i > 0 else ''
        after = sequence[(i + 1):(i + 2)]
        if is_space(char):
            tokens.append(char)
            editable.append(False)
            can_append = False
        elif is_special_character(char) and not is_special_character_exception(char, before, after):
            tokens.append(char)
            editable.append(False)
            can_append = False
        else:
            if can_append:
                tokens[-1] += char
            else:
                tokens.append(char)
                editable.append(True)
                can_append = True
    return list(zip(editable, tokens))


def split2words(merged, split):
    return [merged[a:b] for (a, b) in split]


def combine_mergable_tokens_all(tokens):
    token_lists = []
    t_i = 0
    while t_i < len(tokens):
        current = tokens[t_i]
        if not current[0]:
            token_lists.append((False, [current[1]]))
        else:
            words_to_merge = [current[1]]
            while t_i < len(tokens) - 2:
                if tokens[t_i + 1][1] == ' ' and tokens[t_i + 2][0]:
                    words_to_merge.append(tokens[t_i + 2][1])
                    t_i += 2
                else:
                    break
            token_lists.append((True, words_to_merge))
        t_i += 1
    return token_lists


def get_word_positions(string, word_dict, max_word_len=None):
    positions = []
    if max_word_len is None:
        max_word_len = len(string)
    for i in range(len(string)):
        end = min(len(string), i + max_word_len) + 1
        for j in range(i + 1, end):
            if is_word(string[i:j], word_dict):
                positions.append((i, j))
    return positions


def space_corruption_positions(original: str,
                               transformed: str):
    insertion_positions = set()
    deletion_positions = set()
    original_pos = 0
    transformed_pos = 0
    while original_pos < len(original) or transformed_pos < len(transformed):
        if original_pos < len(original) and transformed_pos < len(transformed) \
                and original[original_pos] == transformed[transformed_pos]:
            original_pos += 1
            transformed_pos += 1
        elif original_pos < len(original) and original[original_pos] == ' ':
            deletion_positions.add(original_pos)
            original_pos += 1
        elif transformed_pos < len(transformed) and transformed[transformed_pos] == ' ':
            insertion_positions.add(original_pos)
            transformed_pos += 1
        else:
            raise Exception("Sequences differ in character unequal to space.")
    return insertion_positions, deletion_positions
