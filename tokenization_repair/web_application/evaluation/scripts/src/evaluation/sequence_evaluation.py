
from src.sequence.predicted_sequence import PredictedSequence
from src.evaluation.samples import get_space_corruptions
from src.sequence.corruption import invert_corruption


def _filter_inverse_predictions(prob_prediction_pairs):
    predictions = set([pair[1] for pair in prob_prediction_pairs])
    filtered = []
    for probability, prediction in prob_prediction_pairs:
        if invert_corruption(prediction) not in predictions:
            filtered.append((probability, prediction))
    return filtered


class SequenceEvaluation:
    def __init__(self,
                 predicted_sequence: PredictedSequence,
                 correct: str,
                 corrupt: str):
        self.predicted_sequence = predicted_sequence
        self.correct = correct
        self.corrupt = corrupt

    def evaluate(self, consider_order, filter_inverse):
        prob_prediction_pairs = self.predicted_sequence.get_prob_prediction_pairs(consider_order=consider_order)

        if filter_inverse:
            prob_prediction_pairs = _filter_inverse_predictions(prob_prediction_pairs)

        probabilities = {prediction: probability for probability, prediction in prob_prediction_pairs}

        true_corruptions = self.get_ground_truth()
        predictions = set([pair[1] for pair in prob_prediction_pairs])

        print("- true positives -")
        true_positives = set(true_corruptions).intersection(set(predictions))
        for tp in true_positives:
            print(tp, probabilities[tp])

        print("- false positives -")
        false_positives = predictions.difference(true_corruptions)
        for fp in false_positives:
            print(fp, probabilities[fp])

        print("- false negatives -")
        false_negatives = true_corruptions.difference(predictions)
        for fn in false_negatives:
            print(fn)

    def get_ground_truth(self):
        return set(get_space_corruptions(self.correct, self.corrupt))

    def get_ordered_prob_prediction_pairs(self):
        return self.predicted_sequence.get_prob_prediction_pairs(consider_order=True)
