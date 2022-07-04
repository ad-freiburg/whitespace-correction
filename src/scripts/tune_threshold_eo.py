import argparse
import math
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from whitespace_correction.model import transformer
from whitespace_correction.utils import config, inference, tokenization_repair, common, metrics


def evaluate_thresholds(
        logits: torch.Tensor,
        logits_lengths: List[int],
        sequences: List[Tuple[str, str]],
        temperature: float,
        t1: float,
        t2: float,
        metric: str
) -> float:
    temperatures = torch.tensor([temperature] * len(logits), dtype=torch.float)
    thresholds = torch.tensor([0.5, t1, t2], dtype=torch.float).unsqueeze(0).repeat((len(logits), 1))
    defaults = torch.zeros(len(logits), dtype=torch.long)
    predictions = inference.class_predictions_from_logits(
        logits,
        temperatures=temperatures,
        thresholds_and_defaults=(thresholds, defaults)
    )
    logit_lengths_cumsum = np.cumsum(logits_lengths)
    predictions = [
        predictions[lower:lower + length]
        for lower, length in zip(np.concatenate([[0], logit_lengths_cumsum[:-1]]), logits_lengths)
    ]

    input_sequences = []
    repaired_sequences = []
    target_sequences = []
    for pred, (corrupt_sequence, target_sequence) in zip(predictions, sequences):
        repaired_sequence = tokenization_repair.repair_whitespace(corrupt_sequence, pred)
        input_sequences.append(corrupt_sequence)
        repaired_sequences.append(repaired_sequence)
        target_sequences.append(target_sequence)

    # get metric, larger --> better
    if metric == "sequence_accuracy":
        m = metrics.sequence_accuracy(repaired_sequences, target_sequences)
    elif metric == "tok_rep_f1":
        m = metrics.whitespace_correction_f1_prec_rec(repaired_sequences, target_sequences, input_sequences)[0]
    elif metric == "mned":
        m = -metrics.mean_normalized_sequence_edit_distance(repaired_sequences, target_sequences)
    else:
        raise RuntimeError(f"Unsupported metric {metric}")
    return m


def tune(args: argparse.Namespace) -> None:
    logger = common.get_logger("EO_THRESHOLD_TUNING")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device(args.device)

    cfg: config.Config = config.Config.from_yaml(os.path.join(args.experiment, "config.yaml"))
    model = transformer.get_model_from_config(config=cfg.model,
                                              device=device)
    model = model.eval()
    assert cfg.model.type == "encoder_with_head"

    temperature = 1.0
    if os.path.exists(os.path.join(args.experiment, "temperature.pkl")):
        with open(os.path.join(args.experiment, "temperature.pkl"), "rb") as tf:
            temperature = pickle.load(tf)
        logger.info(f"Found temperature file: setting temperature to {temperature}")

    temperature_no_spaces = 1.0
    if os.path.exists(os.path.join(args.experiment, "temperature_no_spaces.pkl")):
        with open(os.path.join(args.experiment, "temperature_no_spaces.pkl"), "rb") as tf:
            temperature_no_spaces = pickle.load(tf)
        logger.info(f"Found temperature (no spaces) file: setting temperature (no spaces) to {temperature_no_spaces}")

    min_threshold = args.min_threshold if args.min_threshold is not None else args.threshold_step
    thresholds = [round(min_threshold, 3)]
    while round(thresholds[-1] + args.threshold_step, 3) < 1:
        thresholds.append(round(thresholds[-1] + args.threshold_step, 3))

    best_thresholds_no_spaces = {}
    best_thresholds = {}
    for in_dir in args.in_dirs:
        corrupt_file = os.path.join(in_dir, "corrupt.txt")
        correct_file = os.path.join(in_dir, "correct.txt")

        all_outputs = []
        sequences = []

        with open(correct_file, "r") as correct_f, open(corrupt_file, "r") as corrupt_f:
            for correct_line, corrupt_line in zip(correct_f, corrupt_f):
                correct_line = correct_line.strip()
                corrupt_line = corrupt_line.strip()
                if (
                        correct_line == "" or corrupt_line == ""
                        or len(corrupt_line) > cfg.model.encoder.max_num_embeddings - 2
                ):
                    continue

                sequences.append((corrupt_line, correct_line))

        sequences = sorted(sequences, key=lambda e: len(e[0]), reverse=True)

        no_spaces = all(" " not in corrupt_line for corrupt_line, _ in sequences)
        if no_spaces:
            logger.info(f"Directory {in_dir} contains no spaces")

        for i in tqdm(
                range(0, len(sequences), args.batch_size),
                desc=f"Processing directory {in_dir}",
                total=math.ceil(len(sequences) / args.batch_size),
                leave=False,
                disable=not args.show_progress
        ):
            batch = sequences[i: i + args.batch_size]
            corrupt_sequences = [b[0] for b in batch]

            inference_results = model.inference(
                corrupt_sequences
            )
            assert all(isinstance(ir, inference.SequenceClassificationInferenceResult) for ir in inference_results)
            assert len(inference_results[0].logits[0]) == 3
            for ir in inference_results:
                all_outputs.append(torch.tensor(ir.logits, dtype=torch.float)[1:-1, ...])

        output_lengths = [len(output) for output in all_outputs]
        all_outputs = torch.cat(all_outputs).cpu()

        curr_best_thresholds = []
        best_metric = -float("inf")
        for t1 in tqdm(thresholds,
                       desc="Iterating over insertion thresholds",
                       leave=False,
                       disable=not args.show_progress):
            for t2 in tqdm(thresholds,
                           desc="Iterating over deletion thresholds",
                           leave=False,
                           disable=not args.show_progress):
                m = evaluate_thresholds(
                    all_outputs,
                    output_lengths,
                    sequences,
                    temperature_no_spaces if no_spaces else temperature,
                    t1,
                    t2,
                    args.metric
                )
                logger.debug(f"Got {args.metric}={m} for thresholds {(0.5, t1, t2)}")
                if m > best_metric:
                    best_metric = m
                    curr_best_thresholds = [(0.5, t1, t2)]
                elif m == best_metric:
                    curr_best_thresholds.append((0.5, t1, t2))
                    # if some thresholds reach the same score according to the metric, take
                    # the on average more restrictive (higher) ones
                    # best_t1, best_t2 = best_thresholds[in_dir][1:]
                    # if (t1 + t2) / 2 > (best_t1 + best_t2) / 2:
                    #     best_thresholds[in_dir] = (0.5, t1, t2)
                    #     logger.debug(f"Replaced current best thresholds {(best_t1, best_t2)} with {(t1, t2)}, "
                    #                  f"because they reached the same score ({args.metric}={m}), but "
                    #                  f"the last ones are more restrictive (higher) on average")

        logger.info(f"Found {len(curr_best_thresholds)} best thresholds: {curr_best_thresholds}")
        curr_best_avg_threshold = list(np.array(curr_best_thresholds).mean(0))
        if no_spaces:
            best_thresholds_no_spaces[in_dir] = curr_best_avg_threshold
        else:
            best_thresholds[in_dir] = curr_best_avg_threshold
        logger.info(f"Taking average of those as best threshold for {in_dir} (no_spaces={no_spaces}): "
                    f"{curr_best_avg_threshold}")

    avg_best_thresholds = list(np.array(list(best_thresholds.values())).mean(0))
    logger.info(f"Average of best thresholds (with spaces): {avg_best_thresholds}")

    out_file = os.path.join(args.experiment, "thresholds_and_default.pkl")
    with open(out_file, "wb") as of:
        pickle.dump((avg_best_thresholds, 0), of)

    logger.info(f"Saved average of best thresholds on data with spaces in {out_file}")

    avg_best_thresholds_no_spaces = list(np.array(list(best_thresholds_no_spaces.values())).mean(0))
    logger.info(f"Average of best thresholds (no spaces): {avg_best_thresholds_no_spaces}")

    out_file = os.path.join(args.experiment, "thresholds_and_default_no_spaces.pkl")
    with open(out_file, "wb") as of:
        pickle.dump((avg_best_thresholds_no_spaces, 0), of)

    logger.info(f"Saved average of best thresholds on data without spaces in {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Path to an experiment directory")
    parser.add_argument("--in-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)

    # threshold specific args
    parser.add_argument("--metric",
                        choices=["sequence_accuracy", "tok_rep_f1", "mned"], default="tok_rep_f1", required=True)
    parser.add_argument("--threshold-step", type=float, required=True)
    parser.add_argument("--min-threshold", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    tune(parse_args())
