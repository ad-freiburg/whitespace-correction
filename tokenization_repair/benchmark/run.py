import argparse
import os
import pickle
import time
from typing import Any

from tabulate import tabulate

import torch
from torch import nn

from tqdm.auto import tqdm

from trt.model import transformer
from trt.utils import common, config, io
from trt.utils.tokenization_repair import repair_whitespace


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, required=True, help="Path to an experiment directory")
    parser.add_argument("-ca", "--checkpoint_averaging", type=int, default=None,
                        help="Use checkpoint averaging with the n last checkpoints")

    parser.add_argument("-bs", "--batch-size", type=int, default=1,
                        help="Batch size for batch inference")
    parser.add_argument("-sb", "--smart-batching", action="store_true")
    parser.add_argument("-im", "--inference-method", choices={"greedy", "sample", "beam"},
                        default="greedy", help="Inference method to use")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Specifies the beam width to use with the beam inference method")
    parser.add_argument("--sample-topk", type=int, default=5,
                        help="Specifies the topk indices with the highest probabilities "
                             "to sample from with the sample inference method")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("-tr", "--tokenization-repair", action="store_true",
                        help="Whether this benchmark is a tokenization repair benchmark")

    parser.add_argument("-sp", "--show-progress", action="store_true")

    parser.add_argument("--benchmarks", type=str, nargs="+", required=True,
                        help="List of paths to plain text files where lines are sequences.")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output dir where predictions will be stored.")
    parser.add_argument("--save-markdown-dir", type=str, default=None)
    return parser.parse_args()


def run_benchmark(input_file: str,
                  output_file: str,
                  model: nn.Module,
                  config: config.ModelConfig,
                  batch_size: int,
                  is_tokenization_repair: bool,
                  smart_batching: bool,
                  show_progress: bool = False,
                  **kwargs: Any):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model.eval()

    with open(output_file, "w", encoding="utf8") as of, open(input_file, "r", encoding="utf8") as f:
        lines = f.readlines()

        lines = [line.strip() for line in lines]

        if smart_batching:
            indices_lengths = sorted([(i, len(line)) for i, line in enumerate(lines)], key=lambda t: -t[1])
            reordered_lines = [lines[idx] for idx, _ in indices_lengths]
            batches = [reordered_lines[i: i + batch_size] for i in range(0, len(reordered_lines), batch_size)]
        else:
            batches = [lines[i: i + batch_size] for i in range(0, len(lines), batch_size)]

        all_predicted_sequences = []

        for batch in tqdm(batches, disable=not show_progress):

            if config.type == "transformer":
                result = model.inference(sequences=batch, **kwargs)

                if isinstance(result[0], list):
                    result = [irs[0] for irs in result]

                predicted_sequences = [model.decoder.tokenizer.decode(ir.token_ids)
                                       for ir in result]

                if is_tokenization_repair:
                    predicted_sequences = [repair_whitespace(sequence=sequence,
                                                             repair_sequence=repair_sequence)
                                           for sequence, repair_sequence in zip(batch, predicted_sequences)]

            elif config.type == "encoder_with_head":
                kwargs["no_spaces"] = [" " not in seq for seq in batch]
                inference_results = model.inference(sequences=batch, **kwargs)

                if is_tokenization_repair:
                    predicted_sequences = [repair_whitespace(sequence=sequence,
                                                             repair_sequence=ir.predictions[1:-1])
                                           for sequence, ir in zip(batch, inference_results)]
                else:
                    predictions = [ir.predictions for ir in inference_results]
                    predicted_sequences = [str(p) for p in predictions]

            else:
                raise NotImplementedError()

            all_predicted_sequences.extend(predicted_sequences)

            yield batch, predicted_sequences

        if smart_batching:
            reordered_predicted_sequences = [None] * len(indices_lengths)

            for i, (idx, _) in enumerate(indices_lengths):
                reordered_predicted_sequences[idx] = all_predicted_sequences[i]

            all_predicted_sequences = reordered_predicted_sequences

        of.write("\n".join(all_predicted_sequences) + "\n")


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("BENCHMARK")

    device = torch.device(args.device)

    cfg: config.Config = config.Config.from_yaml(os.path.join(args.experiment, "config.yaml"))
    model = transformer.get_model_from_config(config=cfg.model,
                                              device=device)

    if args.checkpoint_averaging:
        last_n_checkpoints = io.last_n_checkpoints(os.path.join(args.experiment, "checkpoints"),
                                                   args.checkpoint_averaging)
        checkpoint = io.load_averaged_checkpoint(last_n_checkpoints)

    else:
        checkpoint_path = io.glob_safe(os.path.join(args.experiment, "checkpoints", "*-checkpoint-best.pt"))[0]
        checkpoint = io.load_checkpoint(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    kwargs = {
        "beam_width": args.beam_width,
        "sample_top_k": args.sample_topk,
        "temperature": 1.0,
        "temperature_no_spaces": 1.0,
        "thresholds_and_default": None,
        "thresholds_and_default_no_spaces": None
    }

    if os.path.exists(os.path.join(args.experiment, "temperature.pkl")):
        with open(os.path.join(args.experiment, "temperature.pkl"), "rb") as tf:
            kwargs["temperature"] = pickle.load(tf)
        logger.info(f"Found temperature file: setting temperature to {kwargs['temperature']}")
    if os.path.exists(os.path.join(args.experiment, "temperature_no_spaces.pkl")):
        with open(os.path.join(args.experiment, "temperature_no_spaces.pkl"), "rb") as tf:
            kwargs["temperature_no_spaces"] = pickle.load(tf)
        logger.info(f"Found temperature (no spaces) file: setting temperature to {kwargs['temperature_no_spaces']}")
    if os.path.exists(os.path.join(args.experiment, "thresholds_and_default.pkl")):
        with open(os.path.join(args.experiment, "thresholds_and_default.pkl"), "rb") as tf:
            kwargs["thresholds_and_default"] = pickle.load(tf)
        logger.info(f"Found thresholds_and_default file: setting thresholds and default to "
                    f"{kwargs['thresholds_and_default']}")
    if os.path.exists(os.path.join(args.experiment, "thresholds_and_default_no_spaces.pkl")):
        with open(os.path.join(args.experiment, "thresholds_and_default_no_spaces.pkl"), "rb") as tf:
            kwargs["thresholds_and_default_no_spaces"] = pickle.load(tf)
        logger.info(f"Found thresholds_and_default (no spaces) file: setting thresholds and default to "
                    f"{kwargs['thresholds_and_default_no_spaces']}")

    runtimes = []

    for benchmark_file in args.benchmarks:
        benchmark_path = benchmark_file.split("/")
        assert len(benchmark_path) >= 3 and benchmark_path[-1] == "corrupt.txt", \
            "expected benchmarks to point to files with paths of the format" \
            "some_path/<benchmark_name>/<benchmark_split>/corrupt.txt"
        benchmark_name = benchmark_path[-3]
        benchmark_split = benchmark_path[-2]

        logger.info(f"[{args.model_name}] Running benchmark {benchmark_name} {benchmark_split}")

        output_dir = os.path.join(args.output_dir, benchmark_name, benchmark_split)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.model_name}.txt")

        start = time.monotonic()
        for _ in run_benchmark(input_file=benchmark_file, output_file=output_file, model=model, config=cfg.model,
                               batch_size=args.batch_size, is_tokenization_repair=args.tokenization_repair,
                               smart_batching=args.smart_batching, show_progress=args.show_progress, **kwargs):
            pass
        runtime = time.monotonic() - start

        file_size = os.path.getsize(benchmark_file) / 1024
        file_length = io.line_count(benchmark_file)

        runtimes.append([benchmark_name, benchmark_split, runtime, file_length, file_size])

        logger.info(
            f"[{args.model_name}] Finished benchmark {benchmark_name} {benchmark_split} in {runtime:.2f} seconds")

    runtimes_table = tabulate(runtimes,
                              headers=["Benchmark", "Split", "Runtime in seconds", "Number of samples",
                                       "File size in KB"],
                              tablefmt="pipe")

    total_samples = sum([r[3] for r in runtimes])
    total_file_size = sum([r[4] for r in runtimes])
    total_time = sum([r[2] for r in runtimes])

    aggregated_runtimes = [[args.model_name, total_time, total_samples / total_time, total_time / total_file_size]]

    runtimes_aggregated_table = tabulate(aggregated_runtimes,
                                         headers=["Model", "Total runtime in seconds", "samples/s", "s/KB"],
                                         tablefmt="pipe")

    logger.info(f"\nModel: {args.model_name}, Method: {args.inference_method}, Batch size: {args.batch_size}\n"
                f"{runtimes_table}\n\n"
                f"{runtimes_aggregated_table}\n")

    if args.save_markdown_dir is not None:
        os.makedirs(args.save_markdown_dir, exist_ok=True)

        with open(os.path.join(args.save_markdown_dir,
                               f"{args.model_name}_{args.inference_method}_{args.batch_size}"
                               f"{'_smart_batching' if args.smart_batching else ''}.md"),
                  "w",
                  encoding="utf8") as f:
            f.write(runtimes_table)

            f.write("\n\n")

            f.write(runtimes_aggregated_table)
