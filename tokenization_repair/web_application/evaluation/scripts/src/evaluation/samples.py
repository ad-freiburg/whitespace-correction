from src.sequence.corruption import Corruption, CorruptionType


def corrupt_linebreaks_to_spaces(original, corrupt):
    """
    Transforms corrupt line breaks to corrupt space insertions.

    :param original: The unchanged sequence.
    :param corrupt: The corrupt sequence with line breaks.
    :return: Ground truth sequence, corrupt sequence.
    """
    o_i = 0
    c_i = 0
    original_space_only = ""
    corrupt_space_only = ""
    while o_i < len(original) or c_i < len(corrupt):
        if corrupt[c_i:].startswith("-\n") and not original[o_i:].startswith("-\n"):
            # inserted line break
            corrupt_space_only += ' '
            c_i += 2
        elif o_i < len(original) and c_i < len(corrupt) and original[o_i] == corrupt[c_i]:
            # no change
            corrupt_space_only += original[o_i]
            original_space_only += original[o_i]
            o_i += 1
            c_i += 1
        elif c_i < len(corrupt) and corrupt[c_i] == ' ':
            # inserted space
            corrupt_space_only += ' '
            c_i += 1
        elif o_i < len(original) and original[o_i] in " \n":
            # deleted space or line break
            original_space_only += ' '
            o_i += 1
        else:
            print("**original**\n" + original[(o_i - 20):(o_i + 20)])
            print("**corrupt**\n" + corrupt[(c_i - 20):(c_i + 20)])
            raise Exception("Something went wrong at positions %i (original) and %i (corrupt)." % (o_i, c_i))
    return original_space_only, corrupt_space_only


def get_space_corruptions(original, corrupt, ignore_other_insertions=False):
    o_i = 0
    c_i = 0
    corruptions = []
    while o_i < len(original) or c_i < len(corrupt):
        if o_i < len(original) and c_i < len(corrupt) and original[o_i] == corrupt[c_i]:
            # no corruption
            o_i += 1
            c_i += 1
        elif o_i < len(original) and original[o_i] == ' ':
            # space deleted
            corruptions.append(Corruption(CorruptionType.DELETION, c_i, ' '))
            o_i += 1
        elif c_i < len(corrupt) and corrupt[c_i] == ' ':
            # space inserted
            corruptions.append(Corruption(CorruptionType.INSERTION, c_i, ' '))
            c_i += 1
        elif ignore_other_insertions:
            c_i += 1
        else:
            raise Exception("Original and corrupt sequence differ at character inequal to space, " +
                            "expected only space corruptions." +
                            ("\noriginal: %s" % original) +
                            ("\ncorrupt: %s" % corrupt))
    return corruptions
