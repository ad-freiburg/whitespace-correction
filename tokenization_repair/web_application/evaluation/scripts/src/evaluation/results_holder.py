from enum import Enum

from src.helper.pickle import load_object, dump_object
from src.settings import paths
from src.helper.files import file_exists


class Metric(Enum):
    F1 = 0
    SEQUENCE_ACCURACY = 1
    MEAN_RUNTIME = 2


class ResultsHolder:
    def __init__(self):
        if file_exists(paths.RESULTS_DICT):
            self.results = load_object(paths.RESULTS_DICT)
        else:
            self.results = {}

    def save(self):
        dump_object(self.results, paths.RESULTS_DICT)

    def set(self, benchmark_name, benchmark_subset, approach_name, metric_value_pairs):
        if benchmark_name not in self.results:
            self.results[benchmark_name] = {}
        if benchmark_subset not in self.results[benchmark_name]:
            self.results[benchmark_name][benchmark_subset] = {}
        if approach_name not in self.results[benchmark_name][benchmark_subset]:
            self.results[benchmark_name][benchmark_subset][approach_name] = {}
        for metric, value in metric_value_pairs:
            self.results[benchmark_name][benchmark_subset][approach_name][metric] = value

    def get(self, benchmark_name, benchmark_subset, approach_name, metric):
        return self.results[benchmark_name][benchmark_subset][approach_name][metric]

    def contains(self, benchmark_name, benchmark_subset, approach_name, metric):
        if approach_name not in self.results[benchmark_name][benchmark_subset] or \
                metric not in self.results[benchmark_name][benchmark_subset][approach_name]:
            return False
        return True
