import argparse
import os

from src.benchmark.benchmark import Benchmark, BenchmarkFiles
from src.benchmark.subset import Subset
from src.evaluation.evaluator import Evaluator
from src.evaluation.tolerant import tolerant_preprocess_sequences
from src.evaluation.print_methods import print_evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--benchmark-name", type=str, required=True)
    parser.add_argument("--subset", choices=[s.name for s in Subset], required=True)
    parser.add_argument("--predictions", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    subset = Subset[args.subset].value
    benchmark_path = os.path.join(args.benchmark_dir, args.benchmark_name, subset)
    results_path = os.path.join(args.results_dir, args.benchmark_name, subset)
    assert os.path.exists(benchmark_path), f"could not find {benchmark_path}"

    benchmark = Benchmark(
        benchmark_dir=args.benchmark_dir,
        results_dir=args.results_dir,
        name=args.benchmark_name,
        subset=Subset[args.subset]
    )
    correct_sequences = benchmark.get_sequences(BenchmarkFiles.CORRECT)
    corrupt_sequences = benchmark.get_sequences(BenchmarkFiles.CORRUPT)
    predicted_sequences = benchmark.get_predicted_sequences(args.predictions)
    original_sequences = correct_sequences

    evaluator = Evaluator(results_path)
    for seq_id, (original, correct, corrupt, predicted) in \
            enumerate(zip(original_sequences, correct_sequences, corrupt_sequences, predicted_sequences)):

        if benchmark.name == "acl" and original.startswith("#"):
            print(original)
            continue

        correct_processed, corrupt_processed, predicted_processed = \
            tolerant_preprocess_sequences(original, correct, corrupt, predicted)

        evaluator.evaluate(None,
                           None,
                           original_sequence=correct_processed,
                           corrupt_sequence=corrupt_processed,
                           predicted_sequence=predicted_processed,
                           evaluate_ed=False)
        print(original)
        print(corrupt)
        evaluator.print_sequence()
        print()

    print_evaluator(evaluator)

    os.makedirs(results_path, exist_ok=True)
    evaluator.save_json(results_path, args.predictions)
