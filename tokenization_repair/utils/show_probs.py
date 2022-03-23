import argparse
import os
from typing import Optional, List, Dict

import pandas as pd

from trt.model import encoder
from trt.api import get_available_models, TokenizationRepairer
from trt.utils import common, hooks, constants, tokenization_repair

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        choices=[model.name for model in get_available_models()],
        default=get_available_models()[0].name
    )
    parser.add_argument("-r", "--repair", type=str, required=True)
    return parser.parse_args()


def show_probabilities(args: argparse.Namespace) -> None:
    logger = common.get_logger("SHOW_PROBS")
    tok_rep = TokenizationRepairer.from_pretrained(args.model, device="cpu")

    is_encoder_only = args.model.startswith("eo")

    model_hook = hooks.ModelHook()

    logits_hook = hooks.SaveOutputHook()
    if is_encoder_only:
        model_hook.attach(
            "logits", tok_rep.model, logits_hook, nn.Linear, layer_name="head.head.0"
        )
    else:
        model_hook.attach(
            "logits", tok_rep.model, logits_hook, nn.Linear, layer_name="decoder.out_proj"
        )

    repaired_text = tok_rep.repair_text(args.repair)

    if is_encoder_only:
        logits = model_hook["logits"]["head.head.0"][0].squeeze(1)

        characters = [constants.BOS] + list(args.repair.strip()) + [constants.EOS]

        for logits_, char in zip(logits, characters):
            print(char, "\t", [round(f, 2) for f in torch.softmax(logits_, dim=0).tolist()])

    else:
        tokenizer = tok_rep.model.decoder.tokenizer
        logits = model_hook["logits"]["decoder.out_proj"]
        logits = [l[-1, 0, :] for l in logits]
        for l in logits:
            top_k_val, top_k_ind = torch.topk(torch.softmax(l, dim=0), k=3, dim=0)
            print(
                [tokenizer.id_to_token(token_id) for token_id in top_k_ind.tolist()], "\t",
                [round(f, 2) for f in top_k_val.tolist()]
            )

    logger.info(f"Repaired '{args.repair}' to '{repaired_text}'")


if __name__ == "__main__":
    show_probabilities(parse_args())
