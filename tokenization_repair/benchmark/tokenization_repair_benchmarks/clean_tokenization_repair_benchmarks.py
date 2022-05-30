import argparse
import os
from typing import Set

from whitespace_repair.utils import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", type=str, nargs="+", required=True)
    parser.add_argument("--groundtruth", type=str, nargs="+", required=True)
    parser.add_argument("--results", type=str, nargs="+", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--benchmark-out-dir", type=str, required=True)
    parser.add_argument("--results-out-dir", type=str, required=True)
    return parser.parse_args()


def clean_benchmark(corrupted_file: str, groundtruth_file: str, max_length: int, out_dir: str) -> Set[int]:
    os.makedirs(out_dir, exist_ok=True)
    removed_lines = set()

    with open(corrupted_file, "r", encoding="utf8") as cf, \
            open(groundtruth_file, "r", encoding="utf8") as gf, \
            open(os.path.join(out_dir, "corrupt.txt"), "w", encoding="utf8") as ocf, \
            open(os.path.join(out_dir, "correct.txt"), "w", encoding="utf8") as ogf:
        for i, (c_seq, g_seq) in enumerate(zip(cf, gf)):
            c_seq = c_seq.strip()
            g_seq = g_seq.strip()

            if max(len(c_seq), len(g_seq)) > max_length:
                removed_lines.add(i)
                continue

            ocf.write(c_seq + "\n")
            ogf.write(g_seq + "\n")

    return removed_lines


def clean_prediction(prediction_file: str, removed_lines: Set[int], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    model_name = os.path.basename(prediction_file)

    with open(prediction_file, "r", encoding="utf8") as pf, \
            open(os.path.join(out_dir, model_name), "w", encoding="utf8") as opf:
        for i, line in enumerate(pf):
            if i in removed_lines:
                continue

            opf.write(line.strip() + "\n")


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("CLEAN_TOKENIZATION_REPAIR_BENCHMARKS")

    for corrupted_file in args.corrupted:
        corrupted_path = corrupted_file.split("/")
        assert len(corrupted_path) >= 3 and corrupted_path[-1] == "corrupt.txt", \
            "expected corrupted to point to files with paths of the format" \
            " some_path/<benchmark_name>/<benchmark_split>/corrupt.txt"
        benchmark_name = corrupted_path[-3]
        benchmark_split = corrupted_path[-2]

        benchmark_out_dir = os.path.join(args.benchmark_out_dir, benchmark_name, benchmark_split)

        for groundtruth_file in args.groundtruth:
            groundtruth_path = groundtruth_file.split("/")
            assert len(groundtruth_path) >= 3 and groundtruth_path[-1] == "correct.txt", \
                "expected groundtruth to point to files with paths of the format" \
                " some_path/<benchmark_name>/<benchmark_split>/correct.txt"

            if benchmark_name != groundtruth_path[-3] or benchmark_split != groundtruth_path[-2]:
                continue

            removed_lines = clean_benchmark(corrupted_file=corrupted_file,
                                            groundtruth_file=groundtruth_file,
                                            max_length=args.max_length,
                                            out_dir=benchmark_out_dir)

            logger.info(f"Removed {len(removed_lines)} lines longer than {args.max_length} from benchmark "
                        f"{benchmark_name} {benchmark_split}")

            for result_file in args.results:
                result_path = result_file.split("/")
                assert len(result_path) >= 3 and result_path[-1].endswith(".txt"), \
                    "expected prediction to point to files with paths of the format" \
                    " some_path/<benchmark_name>/<benchmark_split>/<model_name>.txt"

                if benchmark_name != result_path[-3] or benchmark_split != result_path[-2]:
                    continue

                results_out_dir = os.path.join(args.results_out_dir, benchmark_name, benchmark_split)

                clean_prediction(prediction_file=result_file,
                                 removed_lines=removed_lines,
                                 out_dir=results_out_dir)

                logger.info(f"Cleaned prediction {os.path.basename(result_file)} for benchmark "
                            f"{benchmark_name} {benchmark_split}")
