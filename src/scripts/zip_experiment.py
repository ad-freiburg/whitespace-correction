import argparse
import glob
import os
import zipfile

import torch

from whitespace_repair.utils import io


def zip_experiment(args: argparse.Namespace) -> None:
    if not args.out_file.endswith(".zip"):
        args.out_file += ".zip"

    with zipfile.ZipFile(args.out_file, "w") as zip_file:
        checkpoint_best = io.glob_safe(os.path.join(args.experiment, "checkpoints", "*-checkpoint-best.pt"))
        assert len(checkpoint_best) == 1
        checkpoint = io.load_checkpoint(checkpoint_best[0])
        only_model_checkpoint = {"model_state_dict": checkpoint["model_state_dict"]}
        only_model_checkpoint_path = os.path.join(args.experiment, "checkpoints", "model-only-checkpoint-best.pt")
        torch.save(only_model_checkpoint, only_model_checkpoint_path)

        experiment_dir = os.path.join(args.experiment, "..")

        config_path = os.path.join(args.experiment, "config.yaml")
        pickle_files = glob.glob(os.path.join(args.experiment, "*.pkl"))

        # best checkpoint
        zip_file.write(
            os.path.join(only_model_checkpoint_path),
            os.path.relpath(only_model_checkpoint_path, experiment_dir)
        )

        # config
        zip_file.write(
            os.path.join(config_path),
            os.path.relpath(config_path, experiment_dir)
        )

        # optional pickle files
        for pickle_file in pickle_files:
            zip_file.write(
                os.path.join(pickle_file),
                os.path.relpath(pickle_file, experiment_dir)
            )

        # delete only model checkpoint again
        os.remove(only_model_checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    zip_experiment(parse_args())
