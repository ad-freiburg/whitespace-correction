import argparse
import collections
import os
from typing import Dict

from tqdm import tqdm

from whitespace_correction.utils import common, metrics, io, tables


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)

    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--save-dir", type=str, required=True)

    return parser.parse_args()


def evaluate(
        groundtruth_file: str,
        predicted_file: str,
        corrupted_file: str
) -> Dict[str, float]:
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

    mic_f1, mic_precision, mic_recall, seq_f1, seq_precision, seq_recall = metrics.whitespace_correction_f1_prec_rec(
        predictions, groundtruths, corrupted
    )
    ins_f1, ins_precision, ins_recall, _, _, _ = metrics.whitespace_correction_f1_prec_rec(
        predictions, groundtruths, corrupted, mode="insertions"
    )
    del_f1, del_precision, del_recall, _, _, _ = metrics.whitespace_correction_f1_prec_rec(
        predictions, groundtruths, corrupted, mode="deletions"
    )
    seq_acc = metrics.sequence_accuracy(predictions, groundtruths)

    return {
        "seq_acc": seq_acc * 100,
        "mic_f1": mic_f1 * 100,
        "seq_avg_f1": seq_f1 * 100,
        "ins_f1": ins_f1 * 100,
        "del_f1": del_f1 * 100
    }


_METRIC_TO_FMT = {
    "seq_acc": ".1f",
    "mic_f1": ".1f",
    "seq_avg_f1": ".1f",
    "ins_f1": ".1f",
    "del_f1": ".1f"
}

if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE_PAPER")

    benchmarks = sorted(io.glob_safe(os.path.join(args.benchmark_dir, "*", "test")))
    benchmark_names = [b.split("/")[-2] for b in benchmarks]

    models = [
        # baselines
        "do_nothing",
        "google",
        "wordsegment",
        # previous work
        "the-one",
        "bid+",
        # ed
        "ed_small",
        "ed_medium",
        "ed_large",
        # encoder only
        "eo_small",
        "eo_medium",
        "eo_large",
    ]

    horizontal_lines = [
        # baselines
        False,
        False,
        True,
        # encoder only
        False,
        True,
        # encoder-decoder
        False,
        False,
        True,
        # previous work
        False,
        False,
        True
    ]

    results = collections.defaultdict(list)

    for model in tqdm(models, total=len(models), desc="evaluating models", leave=False):
        model_scores = collections.defaultdict(list)
        for benchmark, benchmark_name in tqdm(
                zip(benchmarks, benchmark_names), total=len(benchmarks), desc="evaluating benchmarks", leave=False
        ):
            benchmark_input = os.path.join(benchmark, "corrupt.txt")
            benchmark_gt = os.path.join(benchmark, "correct.txt")
            model_prediction = os.path.join(args.result_dir, benchmark_name, "test", f"{model}.txt")
            # for the-one, fallback to bid+ (they are the same on benchmarks with no whitespaces)
            if not os.path.exists(model_prediction) and model == "the-one":
                model_prediction = os.path.join(args.result_dir, benchmark_name, "test", "bid+.txt")

            if not os.path.exists(model_prediction):
                for metric_name in _METRIC_TO_FMT:
                    model_scores[metric_name].append(None)
                continue

            m = evaluate(
                benchmark_gt, model_prediction, benchmark_input
            )

            for metric_name, score in m.items():
                model_scores[metric_name].append(score)

        for metric_name, benchmark_scores in model_scores.items():
            results[metric_name].append(
                [model] + benchmark_scores
            )

    for metric_name, data in results.items():
        if len(data) == 0:
            continue

        best_scores_per_benchmark = [float("-inf")] * len(data[0])
        best_models_per_benchmark = [set()] * len(data[0])
        for i, model_scores_per_benchmark in enumerate(data):
            for j, benchmark_score in enumerate(model_scores_per_benchmark):
                if j == 0:  # j == 0 is model name
                    continue
                if benchmark_score is None:
                    continue
                elif benchmark_score == best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j].add(i)
                elif benchmark_score > best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j] = {i}
                    best_scores_per_benchmark[j] = benchmark_score

        bold_cells = set(
            (i, j) for j, best_models in enumerate(best_models_per_benchmark) for i in best_models
        )

        # convert data to strings
        data = [
            line[:1] + [
                f"{score:{_METRIC_TO_FMT[metric_name]}}" if score is not None else "-"
                for score in line[1:]
            ]
            for line in data
        ]

        results_table = tables.generate_table(
            header=["Model"] + benchmark_names,
            data=data,
            horizontal_lines=horizontal_lines,
            bold_cells=bold_cells,
            fmt=args.format
        )
        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, f"{metric_name}.{'md' if args.format == 'markdown' else 'tex'}")

        with open(path, "w", encoding="utf8") as f:
            f.write(results_table)

        logger.info(f"Results table: {metric_name}\n{results_table}")
