from src.benchmark.benchmark import Benchmark, Subset, BenchmarkFiles, get_benchmark_name


def get_two_pass_benchmark(noise_level: float, p: float, subset: Subset, file_name: str):
    name = get_benchmark_name(noise_level, p)
    return TwoPassBenchmark(name, file_name, subset)


class TwoPassBenchmark(Benchmark):
    def __init__(self, benchmark_name: str, file_name: str, subset: Subset):
        super().__init__(benchmark_name, subset)
        self.file_name = file_name

    def get_sequences(self, file: BenchmarkFiles):
        if file == BenchmarkFiles.CORRUPT:
            return self.get_predicted_sequences(self.file_name)
        else:
            return super().get_sequences(file)

