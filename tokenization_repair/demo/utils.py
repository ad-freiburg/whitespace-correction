import os
from typing import Callable, List, Tuple, Union

import streamlit as st

import tokenizers

import torch

from trt.model.transformer import TransformerDecoderModel, TransformerEncoderModelWithHead, TransformerModel, \
    get_model_from_config
from trt.utils import config, io

BASE_DIR = os.path.dirname(__file__)


def hash_tokenizer(tok: tokenizers.Tokenizer) -> int:
    return id(tok)


@st.cache(show_spinner=False, hash_funcs={tokenizers.Tokenizer: hash_tokenizer})
def load_model(config: config.ModelConfig, path: str, device: torch.device) \
    -> Tuple[Union[TransformerEncoderModelWithHead,
                   TransformerDecoderModel,
                   TransformerModel],
             int]:
    model = get_model_from_config(config=config,
                                  device=device)
    checkpoint = io.load_checkpoint(path)

    io.load_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["step"]


@st.cache(show_spinner=False)
def get_example_sequences() -> List[str]:
    with open(os.path.join(BASE_DIR, "example_sequences.txt"), "r", encoding="utf8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def get_attention_type(layer_name: str) -> str:
    if "decoder" in layer_name:
        if "self_attn" in layer_name:
            attention_type = "Decoder self-attention"
        # decoder-encoder cross attention
        else:
            attention_type = "Decoder-encoder cross-attention"
    # encoder self attention
    else:
        attention_type = "Encoder self-attention"
    return attention_type


def get_embedding_type(layer_name: str) -> str:
    if "decoder" in layer_name:
        if "pos_embedding" in layer_name:
            embedding_type = "Decoder positional embedding"
        else:
            embedding_type = "Decoder token embedding"
    else:
        if "pos_embedding" in layer_name:
            embedding_type = "Encoder positional embedding"
        else:
            embedding_type = "Encoder token embedding"
    return embedding_type


def format_inference_result(sequence: str,
                            predicted_sequence: str,
                            transform_result_fn: Callable[[str, str], str] = None) -> str:
    display_seq = transform_result_fn(sequence, predicted_sequence) \
        if transform_result_fn is not None else predicted_sequence
    return display_seq


def last_n_k_path(path: str, n: int = 3, k: int = None) -> str:
    if k is None:
        return "/".join(path.split("/")[-n:])
    else:
        return "/".join(path.split("/")[-n:-k])
