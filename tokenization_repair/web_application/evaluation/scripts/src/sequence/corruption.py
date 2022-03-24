from enum import Enum


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


def revert_corruption(sequence, corruption):
    if corruption.type == CorruptionType.INSERTION:
        return sequence[:corruption.position] + sequence[(corruption.position + 1):]
    return sequence[:corruption.position] + corruption.character + sequence[corruption.position:]


def invert_corruption(corruption):
    inverse_type = CorruptionType.INSERTION if corruption.type == CorruptionType.DELETION else CorruptionType.DELETION
    return Corruption(inverse_type, corruption.position, corruption.character)
