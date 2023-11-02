import argparse
import os
from typing import Optional, List

import pandas as pd

from whitespace_correction.api import WhitespaceCorrector

from text_utils import hook, logging

from torch import nn
import numpy as np

try:
    import altair as alt
except ImportError:
    raise ImportError("install altair to generate visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "-m",
        "--model",
        choices=[model.name for model in WhitespaceCorrector.available_models()],
        default=WhitespaceCorrector.default_model().name,
        help=f"Name of the model to use for {WhitespaceCorrector.task}"
    )
    model_group.add_argument(
        "-e",
        "--experiment",
        type=str,
        default=None,
        help="Path to an experiment directory from which the model will be loaded "
             "(use this when you trained your own model and want to use it)"
    )
    parser.add_argument("-l", "--layers", type=int, nargs="+", required=True)
    parser.add_argument("-c", "--correct", type=str, required=True)
    parser.add_argument("-s", "--save-dir", type=str, required=True)
    parser.add_argument(
        "-t",
        "--type",
        choices=[
            "attention",
            "attention_heads",
        ]
    )
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

    tooltips = [
        "x position",
        "y position",
        "x token",
        "y token",
        "weight"
    ]

    df = pd.DataFrame({
        "x": x,
        "y_left": y_left,
        "x position": x_positions,
        "y position": y_positions,
        "x token": repeated_x_labels,
        "y token": repeated_y_labels_left,
        "weight": attention_map.ravel()
    })

    if output_tokens is not None:
        repeated_output_tokens = []
        for i, ot in enumerate(output_tokens):
            for _ in range(T):
                repeated_output_tokens.append(ot)

        df["output token"] = repeated_output_tokens

        tooltips.insert(3, "output token")

    x_axis = alt.X(
        "x:N",
        axis=alt.Axis(
            title=x_label,
            labelExpr="slice(datum.label, 0, -4)",
            labelAngle=-45,
            labelOverlap=False
        ),
        sort=None
    )

    y_axis = alt.Y(
        "y_left:N",
        axis=alt.Axis(
            title=y_label_left,
            labelExpr="slice(datum.label, 0, -4)",
            labelOverlap=False
        ),
        sort=None
    )

    chart = alt.Chart(df, title=name).mark_rect().encode(
        x=x_axis,
        y=y_axis,
        color=alt.Color(
            "weight:Q",
            scale=alt.Scale(scheme="blues")
        ),
        tooltip=tooltips
    ).interactive().configure_axis(grid=False)

    return chart


def generate_visualizations(args: argparse.Namespace) -> None:
    logger = logging.get_logger("GENERATE_ATTENTION_MAPS")
    if args.experiment is not None:
        cor = WhitespaceCorrector.from_experiment(args.experiment, device="cpu")
    else:
        cor = WhitespaceCorrector.from_pretrained(args.model, device="cpu")
    cor.set_precision("fp32")

    model_hook = hook.ModelHook()

    attention_weights_hook = hook.AttentionWeightsHook()
    model_hook.attach(
        "attention_weights",
        cor.model,
        attention_weights_hook,
        nn.MultiheadAttention
    )

    corrected = cor.correct_text(args.correct)
    logger.info(f"Corrected '{args.correct}' to '{corrected}'")

    tokenized = cor.input_tokenizer.tokenize(args.correct)
    prefix = [
        cor.input_tokenizer.id_to_special_token(tok_id)
        for tok_id in tokenized.token_ids[:cor.input_tokenizer.num_prefix_tokens()]
    ]
    suffix = [
        cor.input_tokenizer.id_to_special_token(tok_id)
        for tok_id in tokenized.token_ids[-cor.input_tokenizer.num_suffix_tokens():]
    ]
    characters = prefix + list(args.correct) + suffix

    os.makedirs(args.save_dir, exist_ok=True)

    layers = [
        len(model_hook["attention_weights"]) if layer == -1 else layer
        for layer in args.layers
    ]
    layer_names = list(model_hook["attention_weights"].keys())
    all_weights = [
        weights[0][0].numpy() for weights in model_hook["attention_weights"].values()
    ]

    for layer_name, attention_weights in zip(layer_names, all_weights):
        layer = int(layer_name.split(".")[-2]) + 1
        if layer not in layers:
            logger.error(
                f"could not find layer {layer} with name {layer_name} in {layers}"
            )
            continue

        if args.type == "attention":
            logger.info(f"generating attention chart for layer {layer_name}")
            attention_weights = attention_weights.mean(axis=0)
            chart = attention_chart(
                f"Self-attention layer {layer}",
                attention_weights,
                x_labels=characters,
                x_label="Context",
                y_labels_left=characters,
                y_label_left="Input character"
            )
            chart.save(
                os.path.join(args.save_dir, f"{cor.name}_layer_{layer}.png"),
                scale_factor=4
            )
        elif args.type == "attention_heads":
            for head in range(attention_weights.shape[0]):
                logger.info(f"generating attention chart for layer {layer_name} and head {head + 1}")
                chart = attention_chart(
                    f"Self-attention layer {layer} head {head + 1}",
                    attention_weights[head],
                    x_labels=characters,
                    x_label="Context",
                    y_labels_left=characters,
                    y_label_left="Input character"
                )
                chart.save(
                    os.path.join(
                        args.save_dir,
                        f"{cor.name}_layer_{layer}_head_{head + 1}.png"
                    ),
                    scale_factor=4
                )


if __name__ == "__main__":
    generate_visualizations(parse_args())
