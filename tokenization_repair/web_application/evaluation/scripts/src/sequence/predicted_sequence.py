from typing import List, Tuple

from src.sequence.corruption import CorruptionType, Corruption


class PredictedSequence:
    def __init__(self, original: str):
        self.original = original
        self.sequence = original
        self.original_positions = list(range(len(original)))
        self.predicted_corruptions = []
        self.probabilities = []
        self._inserted_original_positions = set()
        self._deleted_original_positions = set()
        self.runtime = 0

    def delete(self, position, probability=None):
        original_position = self.original_positions[position]
        deleted_character = self.sequence[position]
        self.predicted_corruptions.append(Corruption(CorruptionType.INSERTION, original_position, deleted_character))
        self.sequence = self.sequence[:position] + self.sequence[(position + 1):]
        self.original_positions = self.original_positions[:position] + self.original_positions[(position + 1):]
        self._deleted_original_positions.add(original_position)
        if probability is not None:
            self.probabilities.append(probability)
        return self.sequence

    def insert(self, position, character=' ', probability=None):
        original_position = 0 if position == 0 else self.original_positions[position - 1] + 1
        self.predicted_corruptions.append(Corruption(CorruptionType.DELETION, original_position, character))
        self.sequence = self.sequence[:position] + character + self.sequence[position:]
        self.original_positions = self.original_positions[:position] + [original_position] + \
            self.original_positions[position:]
        self._inserted_original_positions.add(original_position)
        if probability is not None:
            self.probabilities.append(probability)
        return self.sequence

    def _filter_inverse_predictions(self, probability_prediction_pairs: List[Tuple[float, Corruption]]):
        filtered = []
        for probability, prediction in probability_prediction_pairs:
            if prediction.type == CorruptionType.INSERTION and prediction.position in self._inserted_original_positions:
                continue
            if prediction.type == CorruptionType.DELETION and prediction.position in self._deleted_original_positions:
                continue
            filtered.append((probability, prediction))
        return filtered

    def get_prob_prediction_pairs(self, consider_order, filter_inverse=False):
        if not consider_order:
            probabilities = self.probabilities
        else:
            min_probability = 1
            probabilities = []
            for p in self.probabilities:
                min_probability = min(min_probability, p)
                probabilities.append(min_probability)
        pairs = list(zip(probabilities, self.predicted_corruptions))
        if filter_inverse:
            pairs = self._filter_inverse_predictions(pairs)
        return pairs

    def prevent_double_insertion(self, insertion_probabilities):
        for i, original_position in enumerate(self.original_positions):
            if original_position in self._inserted_original_positions:
                insertion_probabilities[i] = -1
        return insertion_probabilities

    def prevent_double_deletion(self, deletion_probabilities):
        for i, original_position in enumerate(self.original_positions):
            if original_position in self._deleted_original_positions:
                deletion_probabilities[i] = -1
        return deletion_probabilities

    def set_runtime(self, runtime: float):
        self.runtime = runtime
