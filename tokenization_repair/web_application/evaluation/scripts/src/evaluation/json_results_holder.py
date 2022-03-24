from typing import Dict

import os

from ..helper.json_files import save_json, load_json


class JsonResultsHolder:
    def __init__(self, directory: str):
        self.directory = directory

    def _json_path(self):
        return os.path.join(self.directory, "results.json")

    def _load(self):
        path = self._json_path()
        if os.path.exists(path):
            return load_json(path)
        else:
            return {}

    def _save(self, results: Dict):
        path = self._json_path()
        save_json(results, path)

    def add_result(self, key: str, precision: float, recall: float, f1: float, sequence_accuracy: float):
        results = self._load()
        results[key] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sequence_accuracy": sequence_accuracy
        }
        self._save(results)
