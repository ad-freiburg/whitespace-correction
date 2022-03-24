from ..sequence.corruption import CorruptionType, Corruption


def inverse_of_deletion(deletion):
    return Corruption(CorruptionType.INSERTION, deletion.position - 1, deletion.character)


def filter_prob_prediction_pairs(prob_prediction_pairs):
    probabilities = {prediction: probability for probability, prediction in prob_prediction_pairs}
    predictions = [prediction for _, prediction in prob_prediction_pairs]
    insertions = set([prediction for prediction in predictions if prediction.type == CorruptionType.INSERTION])
    deletions = set([prediction for prediction in predictions if prediction.type == CorruptionType.DELETION])
    filtered = []
    for deletion in deletions:
        insertion = inverse_of_deletion(deletion)
        if insertion in insertions:
            insertions.discard(insertion)
        else:
            filtered.append(deletion)
    filtered += list(insertions)
    pairs = [(probabilities[prediction], prediction) for prediction in filtered]
    return pairs
