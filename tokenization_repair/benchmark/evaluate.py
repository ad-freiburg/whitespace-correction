import argparse
import os
from typing import Tuple

from tabulate import tabulate

from trt.utils import common, metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--groundtruths", type=str, nargs="+", required=True)
    parser.add_argument("--predictions", type=str, nargs="+", required=True)

    parser.add_argument("--save-markdown-dir", type=str, default=None)

    return parser.parse_args()


def evaluate(
        groundtruth_file: str,
        predicted_file: str,
        corrupted_file: str
) -> Tuple[float, ...]:
    groundtruths = []
    predictions = []
    corrupted = []
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        for gt, p, c in zip(gtf, pf, cf):
            groundtruths.append(gt.strip())
            predictions.append(p.strip())
            corrupted.append(c.strip())

    assert len(predictions) == len(groundtruths) and len(groundtruths) == len(corrupted)

    # f1, precision, recall = metrics.f1_prec_rec(predictions, groundtruths)
    mic_f1, mic_precision, mic_recall, mac_f1, mac_precision, mac_recall = metrics.tok_rep_f1_prec_rec(
        predictions, groundtruths, corrupted
    )
    seq_acc = metrics.sequence_accuracy(predictions, groundtruths)
    mned = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
    med = metrics.mean_sequence_edit_distance(predictions, groundtruths)

    return seq_acc, mned, med, mic_f1, mic_precision, mic_recall, mac_f1, mac_precision, mac_recall


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE")

    for groundtruth_file in args.groundtruths:
        groundtruth_path = groundtruth_file.split("/")
        assert len(groundtruth_path) >= 3 and groundtruth_path[-1] == "correct.txt", \
            "expected groundtruths to point to files with paths of the format" \
            " some_path/<benchmark_name>/<benchmark_split>/correct.txt"
        groundtruth_name = groundtruth_path[-3]
        groundtruth_split = groundtruth_path[-2]

        corrupted_file = os.path.join(os.path.dirname(groundtruth_file), "corrupt.txt")

        benchmark_results = []

        for predicted_file in args.predictions:
            predicted_path = predicted_file.split("/")
            assert len(predicted_path) >= 3, "expected predictions to point to files with paths of the format" \
                                             " some_path/<benchmark_name>/<benchmark_split>/<model_name>.txt"

            if predicted_path[-3] != groundtruth_name or predicted_path[-2] != groundtruth_split:
                continue

            model_name, _ = os.path.splitext(predicted_path[-1])

            seq_acc, mned, med, f1_mic, prec_mic, rec_mic, f1_mac, prec_mac, rec_mac = evaluate(
                groundtruth_file, predicted_file, corrupted_file
            )

            benchmark_results.append(
                [model_name, seq_acc, mned, med, f1_mic, prec_mic, rec_mic, f1_mac, prec_mac, rec_mac]
            )

        benchmark_results = sorted(benchmark_results, key=lambda r: r[1], reverse=True)

        results_table = tabulate(
            benchmark_results,
            headers=[
                "Model", "Sequence accuracy", "MNED",
                "MED", "F1_mic", "Precision_mic", "Recall_mic",
                "F1_mac", "Precision_mac", "Recall_mac"
            ],
            tablefmt="pipe"
        )

        logger.info(f"Benchmark: {groundtruth_name}, {groundtruth_split}:\n{results_table}\n")

        if args.save_markdown_dir is not None:
            os.makedirs(args.save_markdown_dir, exist_ok=True)
            path = os.path.join(args.save_markdown_dir, f"{groundtruth_name}_{groundtruth_split}.md")

            with open(path, "w", encoding="utf8") as f:
                f.write(results_table)
