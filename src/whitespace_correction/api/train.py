import os
from typing import Dict, Any

from torch import nn

from text_correction_utils.api.trainer import Trainer
from text_correction_utils import tokenization

from whitespace_correction.model import model_from_config


class WhitespaceCorrectionTrainer(Trainer):
    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        input_tokenizer = tokenization.Tokenizer.from_config(cfg["input_tokenizer"])
        if "output_tokenizer" in cfg:
            output_tokenizer = tokenization.Tokenizer.from_config(cfg["output_tokenizer"])
        else:
            output_tokenizer = None
        return model_from_config(cfg["model"], input_tokenizer, output_tokenizer)


def main():
    parser = WhitespaceCorrectionTrainer.parser("Train whitespace correction", "Train a whitespace correction model")
    args = parser.parse_args()
    work_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    if args.platform == "local":
        WhitespaceCorrectionTrainer.train_local(work_dir, args.experiment, args.config, args.profile)
    else:
        WhitespaceCorrectionTrainer.train_slurm(work_dir, args.experiment, args.config)


if __name__ == "__main__":
    main()
