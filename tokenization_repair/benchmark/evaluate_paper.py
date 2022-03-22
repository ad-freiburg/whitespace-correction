import argparse
import collections
import os
from typing import Tuple, Dict

from tabulate import tabulate
from tqdm import tqdm

from trt.utils import common, metrics, io


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmarks", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)

    parser.add_argument("--save-latex-dir", type=str, default=None)

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

    mic_f1, mic_precision, mic_recall, seq_f1, seq_precision, seq_recall = metrics.tok_rep_f1_prec_rec(
        predictions, groundtruths, corrupted
    )
    ins_f1, ins_precision, ins_recall, _, _, _ = metrics.tok_rep_f1_prec_rec(
        predictions, groundtruths, corrupted, mode="insertions"
    )
    del_f1, del_precision, del_recall, _, _, _ = metrics.tok_rep_f1_prec_rec(
        predictions, groundtruths, corrupted, mode="deletions"
    )
    seq_acc = metrics.sequence_accuracy(predictions, groundtruths)

    return {
        "seq_acc": seq_acc,
        "mic_f1": mic_f1,
        "seq_avg_f1": seq_f1,
        "ins_f1": ins_f1,
        "del_f1": del_f1
    }


_METRIC_TO_FMT = {
    "seq_acc": ".4f",
    "mic_f1": ".4f",
    "seq_avg_f1": ".4f"
}

if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE_PAPER")

    benchmarks = sorted(io.glob_safe(os.path.join(args.benchmarks, "*", "test")))
    benchmark_names = [b.split("/")[-2] for b in benchmarks]

    models = [
        "the-one",
        "eo_large_arxiv_with_errors",
        "eo_medium_arxiv_with_errors",
        "eo_small_arxiv_with_errors",
        "nmt_large_arxiv_with_errors",
        "nmt_medium_arxiv_with_errors",
        "nmt_small_arxiv_with_errors",
        "google",
        "wordsegment"
    ]

    results = collections.defaultdict(list)

    for model in tqdm(models, total=len(models), desc="evaluating models", leave=False):
        model_scores = collections.defaultdict(list)
        for benchmark, benchmark_name in tqdm(
                zip(benchmarks, benchmark_names), total=len(benchmarks), desc="evaluating benchmarks", leave=False
        ):
            benchmark_input = os.path.join(benchmark, "corrupt.txt")
            benchmark_gt = os.path.join(benchmark, "correct.txt")
            model_prediction = os.path.join(args.results, benchmark_name, "test", f"{model}.txt")
            if not os.path.exists(model_prediction) and model == "the-one":
                model_prediction = os.path.join(args.results, benchmark_name, "test", "bid+.txt")

            if not os.path.exists(model_prediction):
                continue

            m = evaluate(
                benchmark_gt, model_prediction, benchmark_input
            )

            for metric_name, score in m.items():
                model_scores[metric_name].append(score)

        for metric_name, benchmark_scores in model_scores.items():
            results[metric_name].append([model] + benchmark_scores)

    for metric_name, res in results.items():
        results_table_md = tabulate(
            res,
            headers=["Model"] + benchmark_names,
            tablefmt="pipe",
            floatfmt=_METRIC_TO_FMT.get(metric_name, ".4f")
        )
        logger.info(f"Results table: {metric_name}\n{results_table_md}")

        if args.save_latex_dir is not None:
            results_table_tex = tabulate(
                res,
                headers=["Model"] + benchmark_names,
                tablefmt="latex",
                floatfmt=_METRIC_TO_FMT.get(metric_name, ".4f")
            )
            os.makedirs(args.save_latex_dir, exist_ok=True)
            path = os.path.join(args.save_latex_dir, f"{metric_name}.tex")

            with open(path, "w", encoding="utf8") as f:
                f.write(results_table_tex)
