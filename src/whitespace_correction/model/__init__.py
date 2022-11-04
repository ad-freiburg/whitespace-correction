from typing import Union, Tuple

import torch
from torch import nn

from whitespace_correction.model.tokenizer import Tokenizer
from whitespace_correction.model.transformer import (
    TransformerEncoderModelWithHead,
    TransformerDecoderModel,
    TransformerModel
)
from whitespace_correction.model.rnn import (
    RNNEncoderModelWithHead
)
from whitespace_correction.utils.config import ModelConfig

Model = Union[
    TransformerEncoderModelWithHead,
    TransformerDecoderModel,
    TransformerModel,
    RNNEncoderModelWithHead
]


def get_model_from_config(config: ModelConfig,
                          device: torch.device) -> Model:
    if config.type == "transformer_encoder_with_head":
        assert config.encoder is not None and config.head is not None
        model = TransformerEncoderModelWithHead(config=config, device=device)

    elif config.type == "transformer_decoder":
        assert config.decoder is not None
        model = TransformerDecoderModel(config=config, device=device)

    elif config.type == "transformer":
        assert config.encoder is not None and config.decoder is not None
        model = TransformerModel(config=config, device=device)

    elif config.type == "rnn_encoder_with_head":
        assert config.encoder is not None and config.head is not None
        model = RNNEncoderModelWithHead(config=config, device=device)

    else:
        raise ValueError(f"Unknown model type {config.type}")

    return model


def get_tokenizers_from_model(model: nn.Module) -> Tuple[Tokenizer, Tokenizer]:
    if hasattr(model, "encoder"):
        encoder_tokenizer = model.encoder.tokenizer
    else:
        encoder_tokenizer = None

    if hasattr(model, "decoder"):
        decoder_tokenizer = model.decoder.tokenizer
    else:
        decoder_tokenizer = None

    assert encoder_tokenizer is not None or decoder_tokenizer is not None, \
        "Could neither find encoder or decoder tokenizer"

    encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else decoder_tokenizer
    decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else encoder_tokenizer

    return encoder_tokenizer, decoder_tokenizer
