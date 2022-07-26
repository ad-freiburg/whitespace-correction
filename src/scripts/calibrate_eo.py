import argparse
import math
import os
import pickle

import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from tqdm import tqdm

from whitespace_correction.model import transformer
from whitespace_correction.utils import config, inference, whitespace_correction, common


def calc_bins(preds, labels):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(preds, labels):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def calibrate(args: argparse.Namespace) -> None:
    logger = common.get_logger("EO_CALIBRATION")

    device = torch.device(args.device)

    cfg: config.Config = config.Config.from_yaml(os.path.join(args.experiment, "config.yaml"))
    model = transformer.get_model_from_config(config=cfg.model,
                                              device=device)
    model = model.eval()
    assert cfg.model.type == "encoder_with_head"

    temperatures_no_spaces = {}
    temperatures = {}
    for in_dir in args.in_dirs:
        corrupt_file = os.path.join(in_dir, "corrupt.txt")
        correct_file = os.path.join(in_dir, "correct.txt")

        all_outputs = []
        all_preds = []
        all_labels = []

        sequences_and_labels = []
        with open(correct_file, "r") as correct_f, open(corrupt_file, "r") as corrupt_f:
            for correct_line, corrupt_line in zip(correct_f, corrupt_f):
                correct_line = correct_line.strip()
                corrupt_line = corrupt_line.strip()
                if (
                        correct_line == "" or corrupt_line == ""
                        or len(corrupt_line) > cfg.model.encoder.max_num_embeddings - 2
                ):
                    continue

                labels = whitespace_correction.get_whitespace_operations(corrupt_line, correct_line)
                sequences_and_labels.append((corrupt_line, labels))

        sequences_and_labels = sorted(sequences_and_labels, key=lambda e: len(e[0]), reverse=True)

        no_spaces = all(" " not in corrupt_line for corrupt_line, _ in sequences_and_labels)
        if no_spaces:
            logger.info(f"Directory {in_dir} contains no spaces")

        for i in tqdm(
                range(0, len(sequences_and_labels), args.batch_size),
                desc=f"Processing directory {in_dir}",
                total=math.ceil(len(sequences_and_labels) / args.batch_size),
                leave=False,
                disable=not args.show_progress
        ):
            batch = sequences_and_labels[i: i + args.batch_size]
            sequences = [b[0] for b in batch]
            labels = [b[1] for b in batch]
            for label in labels:
                all_labels.append(torch.tensor(label, dtype=torch.long))

            inference_results = model.inference(
                sequences
            )
            assert all(isinstance(ir, inference.SequenceClassificationInferenceResult) for ir in inference_results)
            assert len(inference_results[0].logits[0]) == 3
            for ir in inference_results:
                outputs = torch.tensor(ir.logits, dtype=torch.float)[1:-1, ...]
                all_preds.append(torch.softmax(outputs, -1))
                all_outputs.append(outputs)

        all_outputs = torch.cat(all_outputs)
        all_preds = torch.flatten(torch.cat(all_preds))
        all_labels = torch.cat(all_labels)
        all_one_hot_labels = torch.flatten(F.one_hot(all_labels, 3))

        assert len(all_preds) == len(all_one_hot_labels), (all_preds.shape, all_one_hot_labels.shape)

        ece, mce = get_metrics(all_preds, all_one_hot_labels)
        logger.info(f"Before calibration: ExpectedCalibrationError={ece:.4f}, MaximumCalibrationError={mce:.4f}")

        temperature = nn.Parameter(torch.ones(1, device=device))
        optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10_000, line_search_fn="strong_wolfe")

        all_outputs = all_outputs.to(device)
        all_labels = all_labels.to(device)

        def _closure():
            loss = F.cross_entropy(all_outputs / temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(_closure)

        logger.info(f"Optimal temperature: {temperature.item()}")
        if no_spaces:
            temperatures_no_spaces[in_dir] = temperature.item()
        else:
            temperatures[in_dir] = temperature.item()

        calibrated_preds = torch.flatten(torch.cat([
            torch.softmax(o / temperature.item(), dim=-1) for o in all_outputs.cpu().view(-1, 3)
        ]))

        ece, mce = get_metrics(calibrated_preds, all_one_hot_labels)
        logger.info(f"After calibration: ExpectedCalibrationError={ece:.4f}, MaximumCalibrationError={mce:.4f}")

    if len(temperatures) > 0:
        logger.info(f"Optimal temperatures for each benchmark (with spaces): {temperatures}")
        avg_temperature = sum(temperatures.values()) / len(temperatures)
        logger.info(f"Average temperature on data with spaces: {avg_temperature}")

        out_file = os.path.join(args.experiment, "temperature.pkl")
        with open(out_file, "wb") as of:
            pickle.dump(avg_temperature, of)

        logger.info(f"Saved average temperature on data with spaces in {out_file}")

    if len(temperatures_no_spaces) > 0:
        logger.info(f"Optimal temperatures for each benchmark (no spaces): {temperatures_no_spaces}")
        avg_temperature_no_spaces = sum(temperatures_no_spaces.values()) / len(temperatures_no_spaces)
        logger.info(f"Average temperature on data without spaces: {avg_temperature_no_spaces}")

        out_file = os.path.join(args.experiment, "temperature_no_spaces.pkl")
        with open(out_file, "wb") as of:
            pickle.dump(avg_temperature_no_spaces, of)

        logger.info(f"Saved average temperature on data without spaces in {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Path to an experiment directory")
    parser.add_argument("--in-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
