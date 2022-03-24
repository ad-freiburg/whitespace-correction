def get_inserted_position(original, misspelled):
    if len(original) >= len(misspelled):
        return None
    if misspelled[1:] == original:
        return 0
    for i, m in enumerate(misspelled):
        if i == len(original) or m != original[i]:
            return i
    assert False


def get_inserted_nonspace_positions(original, misspelled):
    inserted_positions = set()
    pos = 0
    for orig, missp in zip(original.split(), misspelled.split()):
        inserted = get_inserted_position(orig, missp)
        if inserted is not None:
            inserted_positions.add(pos + inserted)
        pos += len(missp)
    return inserted_positions


def remove_inserted_nonspace_characters(sequence, nonspace_insertions):
    processed = ""
    nonspace_i = 0
    for i, char in enumerate(sequence):
        if (char == ' ' and not processed.endswith(' ')) or (char != ' ' and nonspace_i not in nonspace_insertions):
            processed += char
        if char != ' ':
            nonspace_i += 1
    return processed


def tolerant_preprocess_sequences(original, correct, corrupt, predicted):
    insertions = get_inserted_nonspace_positions(original, correct)
    correct = remove_inserted_nonspace_characters(correct, insertions)
    corrupt = remove_inserted_nonspace_characters(corrupt, insertions)
    predicted = remove_inserted_nonspace_characters(predicted, insertions)
    return correct, corrupt, predicted
