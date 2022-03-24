from src.helper.files import read_lines, write_lines


class PredictionsFileWriter:
    def __init__(self, file: str):
        self.file = file
        self.predicted_sequences = []
        self.runtime = 0

    def load(self):
        lines = read_lines(self.file)
        self.predicted_sequences = lines[:-1]
        self.runtime = float(lines[-1])

    def n_sequences(self):
        return len(self.predicted_sequences)

    def add(self, predicted_sequence: str, runtime: float):
        self.predicted_sequences.append(predicted_sequence)
        self.runtime += runtime

    def save(self):
        lines = self.predicted_sequences + [str(self.runtime)]
        write_lines(self.file, lines)
