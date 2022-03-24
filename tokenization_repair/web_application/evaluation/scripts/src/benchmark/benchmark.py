import os
from enum import Enum

from .subset import Subset
from ..helper.files import read_lines


class BenchmarkFiles(Enum):
    CORRECT = "correct.txt"
    CORRUPT = "corrupt.txt"
    INSERTIONS = "insertions.txt"
    DELETIONS = "deletions.txt"
    ORIGINAL = "original.txt"


class Benchmark:
    def __init__(
            self,
            benchmark_dir: str,
            results_dir: str,
            name: str,
            subset: Subset
    ):
        self.benchmark_dir = benchmark_dir
        self.results_dir = results_dir
        self.name = name
        self.subset = subset

    def _benchmark_directory(self):
        directory = os.path.join(self.benchmark_dir, self.name, self.subset.value)
        os.makedirs(directory, exist_ok=True)
        return directory

    def _results_directory(self):
        directory = os.path.join(self.results_dir, self.name, self.subset.value)
        os.makedirs(directory, exist_ok=True)
        return directory

    def make_directories(self):
        self._benchmark_directory()
        self._results_directory()

    def get_file(self, file: BenchmarkFiles) -> str:
        return os.path.join(self._benchmark_directory(), file.value)

    def get_sequences(self, file: BenchmarkFiles):
        path = self.get_file(file)
        sequences = read_lines(path)
        return sequences

    def get_sequence_pairs(
            self,
            corrupt_file: BenchmarkFiles
    ):
        correct_sequences = self.get_sequences(BenchmarkFiles.CORRECT)
        corrupt_sequences = self.get_sequences(corrupt_file)
        sequence_pairs = list(zip(correct_sequences, corrupt_sequences))
        return sequence_pairs

    def get_results_directory(self):
        return self._results_directory()

    def get_predicted_sequences(self, predicted_file: str):
        directory = self._results_directory()
        lines = read_lines(os.path.join(directory, predicted_file))
        try:
            float(lines[-1])
            sequences = lines[:-1]
        except:
            sequences = lines
        return sequences

    def get_mean_runtime(self, predicted_file: str) -> float:
        lines = self.get_predicted_sequences(predicted_file)
        try:
            runtime = float(lines[-1])
            mean_runtime = runtime / (len(lines) - 1)
        except ValueError:
            mean_runtime = 0
        return mean_runtime
