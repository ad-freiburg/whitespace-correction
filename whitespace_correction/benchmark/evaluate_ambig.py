import argparse
import os

from tqdm import tqdm

from whitespace_correction.utils import common, tables


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark-dir", type=str, required=True)

    parser.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    parser.add_argument("--save-dir", type=str, required=True)

    return parser.parse_args()


def evaluate(
        groundtruth_file: str,
        predicted_file: str
) -> float:
    correct = []

    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf:
        for gt, p in zip(gtf, pf):
            groundtruths = [gt_.strip() for gt_ in gt.strip().split("\t")]
            p = p.strip()
            correct.append(p in groundtruths)

    return sum(correct) / len(correct)


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE_AMBIG")

    models = [
        "eo_large",
        "eo_medium",
        "eo_small",
        "ed_large",
        "ed_medium",
        "ed_small",
        "do_nothing"
    ]

    results = []

    gt_file = os.path.join(args.benchmark_dir, "correct.txt")

    for model in tqdm(models, total=len(models), desc="evaluating models", leave=False):
        pred_file = os.path.join(args.benchmark_dir, "results", f"{model}.txt")
        if not os.path.exists(pred_file):
            logger.warning(f"Prediction for model {model} does not exist, skipping")
            continue

        score = evaluate(gt_file, pred_file)

        results.append([model, score])

    bold_rows = set()
    best_score = float("-inf")
    for i, (_, score) in enumerate(results):
        if score == best_score:
            bold_rows.add(i)
        elif score > best_score:
            best_score = score
            bold_rows = {i}

    results_table = tables.generate_table(
        header=["Model", "Ambig sequence accuracy"],
        data=[[model, f"{score * 100:.1f}"] for model, score in results],
        bold_cells=set((i, 1) for i in bold_rows),
        fmt=args.format
    )
    logger.info(f"Results table:\n{results_table}")

    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, f"ambig.{'md' if args.format == 'markdown' else '.tex'}")

    with open(path, "w", encoding="utf8") as f:
        f.write(results_table)
