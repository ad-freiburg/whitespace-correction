from typing import List, Optional

from src.sequence.corruption import Corruption
from src.evaluation.samples import get_space_corruptions
from src.helper.pickle import load_object, dump_object
from src.benchmark.subset import Subset
from src.settings import paths
from src.helper.files import get_files, read_lines


class Prediction:
    def __init__(self,
                 ground_truth: bool):
        self.ground_truth = ground_truth
        self.predictor_keys = set()

    def add_predictor_key(self, key: str):
        self.predictor_keys.add(key)


class BenchmarkSequence:
    def __init__(self,
                 correct_sequence: str,
                 corrupt_sequence: str):
        self.correct_sequence = correct_sequence
        self.corrupt_sequence = corrupt_sequence
        self.ground_truth_corruptions = get_space_corruptions(correct_sequence, corrupt_sequence)
        self.predictions = {corruption: Prediction(ground_truth=True) for corruption in self.ground_truth_corruptions}
        self.predicted_sequences = {}

    def add_predictions(self, key: str, predicted_sequence: str):
        self.predicted_sequences[key] = predicted_sequence
        predicted_corruptions = get_space_corruptions(predicted_sequence, self.corrupt_sequence)
        for predicted_corruption in predicted_corruptions:
            if predicted_corruption not in self.predictions:
                self.predictions[predicted_corruption] = Prediction(ground_truth=False)
            self.predictions[predicted_corruption].add_predictor_key(key)

    def __len__(self):
        return len(self.corrupt_sequence)


CSV_COLUMN_HEADERS = ["sequence_id",
                      "correct",
                      "corrupt",
                      "position",
                      "type",
                      "prefix",
                      "suffix",
                      "ground_truth"]

WINDOW_SIZE = 20


def csv_line(sequence_id: int,
             correct_sequence,
             corrupt_sequence,
             corruption: Corruption,
             prediction: Prediction,
             keys: List[str]) -> str:
    position = corruption.position
    prefix = corrupt_sequence[max(0, position - WINDOW_SIZE):position]
    suffix = corrupt_sequence[position:(position + WINDOW_SIZE)]
    line = ";".join([str(sequence_id),
                     "\"" + correct_sequence + "\"",
                     "\"" + corrupt_sequence + "\"",
                     str(position),
                     corruption.type.name,
                     "\"" + prefix + "\"",
                     "\"" + suffix + "\"",
                     "TRUE" if prediction.ground_truth else "FALSE"
                     ])
    for key in keys:
        case = ""
        if prediction.ground_truth:
            if key in prediction.predictor_keys:
                case = "TRUE_POSITIVE"
            else:
                case = "FALSE_NEGATIVE"
        elif key in prediction.predictor_keys:
            case = "FALSE_POSITIVE"
        line += ";" + case
    return line


class ExtendedBenchmark:
    def __init__(self,
                 name: str,
                 subset: Subset,
                 subdir: Optional[str]=None,
                 prediction_file_names: Optional[List[str]]=None):
        self.subdir = paths.benchmark_sub_directory(name, subset, subdir)
        self.sequences = []
        self.prediction_file_names = prediction_file_names if prediction_file_names is not None \
            else self.get_all_prediction_files()

    def benchmark_dir(self):
        return paths.BENCHMARKS_DIR + self.subdir

    def initialize(self):
        directory = self.benchmark_dir()
        correct_sequences = read_lines(directory + "correct.txt")
        corrupt_sequences = read_lines(directory + "corrupt.txt")
        self.sequences.extend([
            BenchmarkSequence(correct_sequence, corrupt_sequence)
            for correct_sequence, corrupt_sequence in zip(correct_sequences, corrupt_sequences)
        ])

    def benchmark_file(self) -> str:
        return self.benchmark_dir() + "benchmark.pkl"

    def results_dir(self) -> str:
        return paths.RESULTS_DIR + self.subdir

    def csv_file(self) -> str:
        return self.results_dir() + "predictions.csv"

    def get_all_prediction_files(self) -> List[str]:
        files = get_files(self.results_dir())
        filtered = [file for file in files if file.endswith(".txt")]
        return filtered

    def register_predictions(self):
        for file_name in self.prediction_file_names:
            path = self.results_dir() + file_name
            for s_i, predicted_sequence in enumerate(read_lines(path)):
                self.sequences[s_i].add_predictions(file_name, predicted_sequence)

    def save(self):
        dump_object(self, self.benchmark_file())

    def print_predictions(self):
        for sequence in self.sequences:
            print(sequence.correct_sequence)
            print(sequence.corrupt_sequence)
            for corruption in sequence.predictions:
                prediction = sequence.predictions[corruption]
                print(corruption, prediction.ground_truth, prediction.predictor_keys)

    def csv_header(self):
        return ";".join(CSV_COLUMN_HEADERS + self.prediction_file_names)

    def write_csv(self):
        path = self.csv_file()
        file = open(path, "w", encoding="utf8")
        file.write(self.csv_header() + '\n')
        prediction_keys = sorted(self.prediction_file_names)
        for s_i, sequence in enumerate(self.sequences):
            for corruption in sequence.predictions:
                prediction = sequence.predictions[corruption]
                line = csv_line(s_i,
                                sequence.correct_sequence,
                                sequence.corrupt_sequence,
                                corruption,
                                prediction,
                                prediction_keys)
                file.write(line + '\n')
        file.close()
