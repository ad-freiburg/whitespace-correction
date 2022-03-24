import numpy as np

from src.sequence.corruption import CorruptionType


def count_occurences(length, values):
    occurences = np.zeros(length, dtype=int)
    for val in values:
        occurences[val] += 1
    return occurences


def corrupt_positions2predicted_positions(seq_len, predicted_corruptions):
    inserted_positions = count_occurences(seq_len + 1,
                                          [pred.position for pred in predicted_corruptions
                                           if pred.type == CorruptionType.INSERTION])
    deleted_positions = count_occurences(seq_len + 1,
                                         [pred.position for pred in predicted_corruptions
                                          if pred.type == CorruptionType.DELETION])
    predicted_positions = np.arange(seq_len + 1) - np.cumsum(inserted_positions) + np.cumsum(deleted_positions)
    predicted_positions = [max(0, p) for p in predicted_positions]
    return predicted_positions


def sort_by_position(corruptions):
    return sorted(corruptions, key=lambda corruption: corruption.position)


def csv_header():
    return "file;line;position;predicted_prefix;predicted_suffix;probability"


class SequenceResult:
    def __init__(self,
                 file_name,
                 line,
                 original_sequence,
                 corrupt_sequence,
                 predicted_sequence,
                 ground_truth_corruptions,
                 predicted_corruption_probabilities,
                 tp_insertions,
                 fp_insertions,
                 fn_insertions,
                 tp_deletions,
                 fp_deletions,
                 fn_deletions,
                 ed_before,
                 ed_after,
                 runtime):
        self.file_name = file_name,
        self.line = line,
        self.original_sequence = original_sequence
        self.corrupt_sequence = corrupt_sequence
        self.predicted_sequence = predicted_sequence
        self.ground_truth_corruptions = ground_truth_corruptions
        self.predicted_corruption_probabilities = predicted_corruption_probabilities
        self.tp_insertions = tp_insertions
        self.fp_insertions = fp_insertions
        self.fn_insertions = fn_insertions
        self.tp_deletions = tp_deletions
        self.fp_deletions = fp_deletions
        self.fn_deletions = fn_deletions
        self.num_tp_insertions = len(tp_insertions)
        self.num_fp_insertions = len(fp_insertions)
        self.num_fn_insertions = len(fn_insertions)
        self.num_tp_deletions = len(tp_deletions)
        self.num_fp_deletions = len(fp_deletions)
        self.num_fn_deletions = len(fn_deletions)
        self.ed_before = ed_before
        self.ed_after = ed_after
        self.runtime = runtime
        self._set_is_correct()

    def tp(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.tp_insertions
        elif type == CorruptionType.DELETION:
            return self.tp_deletions
        else:
            return self.tp_insertions | self.tp_deletions

    def fp(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.fp_insertions
        elif type == CorruptionType.DELETION:
            return self.fp_deletions
        else:
            return self.fp_insertions | self.fp_deletions

    def fn(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.fn_insertions
        elif type == CorruptionType.DELETION:
            return self.fn_deletions
        else:
            return self.fn_insertions | self.fn_deletions

    def num_tp(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.num_tp_insertions
        elif type == CorruptionType.DELETION:
            return self.num_tp_deletions
        else:
            return self.num_tp_insertions + self.num_tp_deletions

    def num_fp(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.num_fp_insertions
        elif type == CorruptionType.DELETION:
            return self.num_fp_deletions
        else:
            return self.num_fp_insertions + self.num_fp_deletions

    def num_fn(self, type=None):
        if type == CorruptionType.INSERTION:
            return self.num_fn_insertions
        elif type == CorruptionType.DELETION:
            return self.num_fn_deletions
        else:
            return self.num_fn_insertions + self.num_fn_deletions

    def csv_string(self, corruptions, type):
        corruptions = sort_by_position(corruptions)
        corrupt_pos2predicted_pos = corrupt_positions2predicted_positions(
            len(self.corrupt_sequence),
            self.predicted_corruption_probabilities.keys())
        lines = []
        for corruption in corruptions:
            pred_pos = corrupt_pos2predicted_pos[corruption.position]
            if type == CorruptionType.INSERTION:
                before = self.predicted_sequence[max(0, pred_pos - 30 + 1):(pred_pos + 1)]
                after = self.predicted_sequence[(pred_pos + 1):(pred_pos + 31)]
            else:
                before = self.predicted_sequence[max(0, pred_pos - 30):pred_pos]
                after = self.predicted_sequence[pred_pos:(pred_pos + 30)]
            probability = self.predicted_corruption_probabilities[corruption] \
                if corruption in self.predicted_corruption_probabilities \
                else 0
            lines.append("%s;%i;%i;\"%s\";\"%s\";%f" % (
                self.file_name[0],
                self.line[0],
                corruption.position,
                before,
                after,
                probability))
        return '\n'.join(lines)

    def _set_is_correct(self):
        self.correct = self.num_fp() + self.num_fn() == 0

    def is_correct(self):
        return self.correct
