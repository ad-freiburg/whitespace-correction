import argparse
import os
from typing import Optional, List, Dict

import pandas as pd

from trt.model import encoder
from trt.api import get_available_models, TokenizationRepairer
from trt.utils import common, hooks, constants, tokenization_repair

import torch
from torch import nn
import altair as alt
import numpy as np
from sklearn.manifold import TSNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        choices=[model.name for model in get_available_models() if model.name.startswith("eo")],
        default=get_available_models()[0].name
    )
    parser.add_argument("-l", "--layers", type=int, nargs="+", required=True)
    parser.add_argument("-r", "--repair", type=str, required=True)
    parser.add_argument("-s", "--save-dir", type=str, required=True)

    parser.add_argument("-t", "--target", type=str, default=None)
    parser.add_argument("--representations", action="store_true")
    return parser.parse_args()


def attention_chart(
        name: str,
        attention_map: np.ndarray,
        x_labels: List[str],
        y_labels_left: List[str],
        x_label: str,
        y_label_left: str,
        output_tokens: Optional[List[str]] = None
) -> alt.Chart:
    T, S = attention_map.shape

    x = []
    x_positions = []
    repeated_x_labels = []
    for _ in range(S):
        for i, xl in enumerate(x_labels):
            x.append(f"{xl}{i:04}")
            x_positions.append(i)
            repeated_x_labels.append(xl)

    y_left = []
    y_positions = []
    repeated_y_labels_left = []
    for i, yl in enumerate(y_labels_left):
        for _ in range(T):
            y_left.append(f"{yl}{i:04}")
            y_positions.append(i)
            repeated_y_labels_left.append(yl)

    tooltips = ["x position",
                "y position",
                "x token",
                "y token",
                "weight"]

    df = pd.DataFrame({"x": x,
                       "y_left": y_left,
                       "x position": x_positions,
                       "y position": y_positions,
                       "x token": repeated_x_labels,
                       "y token": repeated_y_labels_left,
                       "weight": attention_map.ravel()})

    if output_tokens is not None:
        repeated_output_tokens = []
        for i, ot in enumerate(output_tokens):
            for _ in range(T):
                repeated_output_tokens.append(ot)

        df["output token"] = repeated_output_tokens

        tooltips.insert(3, "output token")

    x_axis = alt.X("x:N",
                   axis=alt.Axis(title=x_label,
                                 labelExpr="slice(datum.label, 0, -4)",
                                 labelAngle=-45,
                                 labelOverlap=False),
                   sort=None)

    y_axis = alt.Y("y_left:N",
                   axis=alt.Axis(title=y_label_left,
                                 labelExpr="slice(datum.label, 0, -4)",
                                 labelOverlap=False),
                   sort=None)

    chart = alt.Chart(df, title=name).mark_rect().encode(x=x_axis,
                                                         y=y_axis,
                                                         color=alt.Color("weight:Q",
                                                                         scale=alt.Scale(scheme="reds")),
                                                         tooltip=tooltips).interactive().configure_axis(grid=False)

    return chart


def encoder_representations_chart(
        encoder_representations: Dict[str, List[torch.Tensor]],
        tokens: List[str],
        labels: List[int]
) -> alt.Chart:
    alt_charts = []

    for i, (layer_name, outputs) in enumerate(encoder_representations.items()):
        outputs = outputs[0].squeeze(1).numpy()
        layer = int(layer_name.split(".")[-1]) + 1

        tsne = TSNE(perplexity=10, learning_rate=10)
        tsne_output = tsne.fit_transform(outputs)
        x, y = tsne_output.T
        df = pd.DataFrame({"x": x,
                           "y": y,
                           "token": tokens,
                           "class": [f"{label}" for label in labels],
                           "position": [f"{i}" for i in range(len(tokens))]})

        alt_base = alt.Chart(df, title=f"Layer {layer}").mark_circle(size=60)
        alt_base = alt_base.encode(x="x",
                                   y="y",
                                   tooltip=["class", "token", "position"],
                                   color="class")

        alt_text = alt_base.mark_text(align="left", baseline="middle", dx=7).encode(text="token")

        alt_chart = (alt_base + alt_text).interactive()
        alt_charts.append(alt_chart)

    hstacked_charts = []
    for i in range(0, len(alt_charts), 2):
        hstacked_charts.append(alt.hconcat(*alt_charts[i:i + 2]))

    vstacked_chart = alt.vconcat(*hstacked_charts).configure_axis(grid=False)
    return vstacked_chart


def generate_visualizations(args: argparse.Namespace) -> None:
    logger = common.get_logger("GENERATE_ATTENTION_MAPS")
    tok_rep = TokenizationRepairer.from_pretrained(args.model, device="cpu")

    model_hook = hooks.ModelHook()

    attention_weights_hook = hooks.AttentionWeightsHook()
    model_hook.attach("attention_weights", tok_rep.model, attention_weights_hook, nn.MultiheadAttention)

    encoder_representations_hook = hooks.SaveOutputHook()
    model_hook.attach(
        "encoder_representations", tok_rep.model, encoder_representations_hook, encoder.TransformerEncoderLayer
    )

    logits_hook = hooks.SaveOutputHook()
    model_hook.attach(
        "logits", tok_rep.model, logits_hook, nn.Linear, layer_name="head.head.0"
    )

    repaired_text = tok_rep.repair_text(args.repair)

    logits = model_hook["logits"]["head.head.0"][0].squeeze(1)

    characters = [constants.BOS] + list(args.repair.strip()) + [constants.EOS]

    for logits_, char in zip(logits, characters):
        print(char, "\t", [round(f, 2) for f in torch.softmax(logits_, dim=0).tolist()])

    os.makedirs(args.save_dir, exist_ok=True)

    layers = [len(model_hook["attention_weights"]) if layer == -1 else layer for layer in args.layers]

    for layer_name, attention_weights in model_hook["attention_weights"].items():
        attention_weights = attention_weights[0][0].numpy()
        layer = int(layer_name.split(".")[-2]) + 1
        if layer not in layers:
            continue

        attention_map = attention_chart(
            f"Self-attention layer {layer}",
            attention_weights,
            x_labels=characters,
            x_label="Input characters",
            y_labels_left=characters,
            y_label_left="Input characters"
        )
        attention_map.save(os.path.join(args.save_dir, f"{args.model}_layer_{layer}.pdf"))

    if args.representations:
        assert args.target
        labels = tokenization_repair.get_whitespace_operations(args.repair.strip(), args.target.strip())
        representations_chart = encoder_representations_chart(
            model_hook["encoder_representations"],
            [f"'{char}'" for char in characters],
            [-1] + labels + [-1]
        )
        representations_chart.save(os.path.join(args.save_dir, f"{args.model}_representations.pdf"))

    logger.info(f"Repaired '{args.repair}' to '{repaired_text}'")


if __name__ == "__main__":
    generate_visualizations(parse_args())
