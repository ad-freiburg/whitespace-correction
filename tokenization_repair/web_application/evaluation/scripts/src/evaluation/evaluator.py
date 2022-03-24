import numpy as np
from termcolor import colored

from ..edit_distance.edit_distance import levenshtein
from ..sequence.corruption import CorruptionType
from ..sequence.functions import map_positions
from ..helper.files import path_exists, make_directory
from ..helper.function import current_time_string

from .json_results_holder import JsonResultsHolder
from .metrics import precision_recall_f1
from .metrics import tp, fp, fn
from .samples import get_space_corruptions
from .sequence_result import SequenceResult, csv_header


def filter_corruptions_by_type(corruptions, type):
    return [corruption for corruption in corruptions if corruption.type == type]


def tp_fp_fn_by_type(ground_truth, predictions, type):
    ground_truth = set(filter_corruptions_by_type(ground_truth, type))
    predictions = set(filter_corruptions_by_type(predictions, type))
    tps = tp(ground_truth, predictions)
    fps = fp(ground_truth, predictions)
    fns = fn(ground_truth, predictions)
    return tps, fps, fns


class Evaluator:
    def __init__(self, out_dir: str):
        self.sequence_results = []
        self.time_string = current_time_string()
        self.out_dir = out_dir

    def evaluate(self,
                 file_name,
                 line,
                 original_sequence,
                 corrupt_sequence,
                 predicted_sequence,
                 ground_truth_corruptions=None,
                 probability_predicted_corruption_pairs=None,
                 runtime=0,
                 evaluate_ed=True):
        # ground truth corruptions
        if ground_truth_corruptions is None:
            ground_truth_corruptions = get_space_corruptions(original_sequence, corrupt_sequence)
        # predicted corruptions
        if probability_predicted_corruption_pairs is None:
            predicted_corruptions = get_space_corruptions(predicted_sequence, corrupt_sequence,
                                                          ignore_other_insertions=False)
            probability_predicted_corruption_pairs = [(1, prediction) for prediction in predicted_corruptions]
        # edit distance
        if evaluate_ed:
            ed_before = levenshtein(original_sequence, corrupt_sequence, substitutions=False)
            ed_after = levenshtein(original_sequence, predicted_sequence, substitutions=False)
        else:
            ed_before = ed_after = 0
        # prediction probability dictionary
        probabilities = {prediction: probability for probability, prediction in probability_predicted_corruption_pairs}
        predicted_corruptions = probabilities.keys()
        # tp, fp and fn sets
        tp_insertions, fp_insertions, fn_insertions = tp_fp_fn_by_type(ground_truth_corruptions, predicted_corruptions,
                                                                       CorruptionType.INSERTION)
        tp_deletions, fp_deletions, fn_deletions = tp_fp_fn_by_type(ground_truth_corruptions, predicted_corruptions,
                                                                    CorruptionType.DELETION)
        # register sequence result
        sequence_result = SequenceResult(
            file_name=file_name,
            line=line,
            original_sequence=original_sequence,
            corrupt_sequence=corrupt_sequence,
            predicted_sequence=predicted_sequence,
            ground_truth_corruptions=ground_truth_corruptions,
            predicted_corruption_probabilities=probabilities,
            tp_insertions=tp_insertions,
            fp_insertions=fp_insertions,
            fn_insertions=fn_insertions,
            tp_deletions=tp_deletions,
            fp_deletions=fp_deletions,
            fn_deletions=fn_deletions,
            ed_before=ed_before,
            ed_after=ed_after,
            runtime=runtime)
        self.sequence_results.append(sequence_result)
        return sequence_result

    def mean_ed_before(self):
        return np.mean([result.ed_before for result in self.sequence_results])

    def mean_ed_after(self):
        return np.mean([result.ed_after for result in self.sequence_results])

    def num_tp(self, type=None):
        return np.sum([result.num_tp(type=type) for result in self.sequence_results])

    def num_fp(self, type=None):
        return np.sum([result.num_fp(type=type) for result in self.sequence_results])

    def num_fn(self, type=None):
        return np.sum([result.num_fn(type=type) for result in self.sequence_results])

    def nums_tp_fp_fn(self, type=None):
        return self.num_tp(type=type), self.num_fp(type=type), self.num_fn(type=type)

    def mean_runtime(self):
        return np.mean([result.runtime for result in self.sequence_results])

    def num_sequences(self):
        return len(self.sequence_results)

    def num_correct_sequences(self):
        return np.sum([result.is_correct() for result in self.sequence_results])

    def sequence_accuracy(self):
        return float(np.mean([result.is_correct() for result in self.sequence_results]))

    def _results_folder(self):
        return self.out_dir

    def _write_csv_file(self, result_set, corruption_type):
        assert (result_set in ["true_positives", "false_positives", "false_negatives"])
        corruption_type_string = "insertions" if corruption_type == CorruptionType.INSERTION else "deletions"
        folder = self._results_folder()
        file_name = result_set + "_" + corruption_type_string + ".csv"
        with open(folder + "/" + file_name, 'w') as file:
            file.write(csv_header() + '\n')
            for sequence_result in self.sequence_results:
                corruptions = None
                if result_set == "true_positives" and corruption_type == CorruptionType.INSERTION:
                    corruptions = sequence_result.tp_insertions
                elif result_set == "true_positives" and corruption_type == CorruptionType.DELETION:
                    corruptions = sequence_result.tp_deletions
                elif result_set == "false_positives" and corruption_type == CorruptionType.INSERTION:
                    corruptions = sequence_result.fp_insertions
                elif result_set == "false_positives" and corruption_type == CorruptionType.DELETION:
                    corruptions = sequence_result.fp_deletions
                elif result_set == "false_negatives" and corruption_type == CorruptionType.INSERTION:
                    corruptions = sequence_result.fn_insertions
                elif result_set == "false_negatives" and corruption_type == CorruptionType.DELETION:
                    corruptions = sequence_result.fn_deletions
                csv_string = sequence_result.csv_string(corruptions, corruption_type)
                if len(csv_string) > 0:
                    file.write(csv_string + '\n')

    def write_predicted_sequence_files(self):
        dir = self._results_folder() + "/predicted/"
        make_directory(dir)
        # empty files
        for sequence_result in self.sequence_results:
            with open(dir + sequence_result.file_name[0], 'w') as file:
                pass
        # write to files
        for sequence_result in self.sequence_results:
            with open(dir + sequence_result.file_name[0], 'a') as file:
                file.write(sequence_result.predicted_sequence + '\n')

    def write_csv_files(self):
        self._write_csv_file("true_positives", CorruptionType.INSERTION)
        self._write_csv_file("false_positives", CorruptionType.INSERTION)
        self._write_csv_file("false_negatives", CorruptionType.INSERTION)
        self._write_csv_file("true_positives", CorruptionType.DELETION)
        self._write_csv_file("false_positives", CorruptionType.DELETION)
        self._write_csv_file("false_negatives", CorruptionType.DELETION)

    def print_sequence(self):
        """
        Prints the most recently evaluated predicted sequence with color codes indicating
        true positives (green), false positives (red) and false negatives (yellow).
        """
        result = self.sequence_results[-1]
        seq_len = len(result.predicted_sequence)
        tp_positions = {tp.position for tp in result.tp()}
        fp_positions = {fp.position for fp in result.fp()}
        fn_positions = {fn.position for fn in result.fn()}
        _, predicted2corrupt = map_positions(result.corrupt_sequence, result.predicted_sequence)
        colors = [None] * seq_len
        on_colors = [None] * seq_len
        for i in range(seq_len):
            if result.predicted_sequence[i] == ' ':
                # insertion
                if predicted2corrupt[i] is None:
                    ref_pos = predicted2corrupt[i + 1] if i + 1 < len(predicted2corrupt) else None
                    if ref_pos in tp_positions:
                        on_colors[i] = "on_green"
                    elif ref_pos in fp_positions:
                        on_colors[i] = "on_red"
                elif predicted2corrupt[i] in fn_positions:
                    on_colors[i] = "on_yellow"
            # elif i + 1 < len(predicted2corrupt) \
            #        and result.predicted_sequence[i + 1] != ' ' \
            #        and predicted2corrupt[i + 1] in fn_positions:
            #    on_colors[i] = "on_yellow"
            elif i + 1 < len(predicted2corrupt) \
                    and result.predicted_sequence[i + 1] != ' ':
                # deletion
                ref_pos = predicted2corrupt[i] + 1
                if ref_pos in tp_positions:
                    colors[i] = colors[i + 1] = "green"
                elif ref_pos in fp_positions:
                    colors[i] = colors[i + 1] = "red"
                elif ref_pos in fn_positions:
                    colors[i] = colors[i + 1] = "yellow"
        print_str = "".join(colored(char, colors[i], on_colors[i]) for i, char in enumerate(result.predicted_sequence))
        print(print_str)

    def f1(self):
        tp = self.num_tp()
        fp = self.num_fp()
        fn = self.num_fn()
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        return f1

    def save_json(self, directory: str, key: str):
        results_holder = JsonResultsHolder(directory)
        tp = self.num_tp()
        fp = self.num_fp()
        fn = self.num_fn()
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        sequence_accuracy = self.sequence_accuracy()
        results_holder.add_result(key, precision, recall, f1, sequence_accuracy)
