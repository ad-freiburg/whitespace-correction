import copy
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules.embedding import Embedding
from text_correction_utils.modules.encoder import Encoder, TransformerEncoder, RNNEncoder, CNNEncoder, GroupingEncoder
from text_correction_utils.modules.head import Head, ClassificationHead, SequenceClassificationHead


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: List[int],
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class EncoderWithHead(Model):
    def __init__(
        self,
        embedding: Embedding,
        encoder: Encoder,
        head: Head,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: List[int],
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb, pos_emb = self.embedding(token_ids)
        enc = self.encoder(emb, lengths, pos_emb, **kwargs)
        output = self.head(enc, **kwargs)
        return output, self.encoder.additional_losses()


def get_model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")
    if model_type == "encoder_with_head":
        embedding = get_embedding_from_config(cfg["embedding"], input_tokenizer)
        encoder = get_encoder_from_config(cfg["encoder"])
        head = get_head_from_config(cfg["head"])
        return EncoderWithHead(embedding, encoder, head)

    else:
        raise ValueError(f"unknown model type {model_type}")


def get_embedding_from_config(cfg: Dict[str, Any], input_tokenizer: tokenization.Tokenizer) -> Embedding:
    return Embedding(
        num_embeddings=input_tokenizer.vocab_size(),
        pad_token_id=input_tokenizer.pad_token_id(),
        **cfg
    )


def get_encoder_from_config(cfg: Dict[str, Any]) -> Encoder:
    cfg = copy.deepcopy(cfg)
    enc_type = cfg.pop("type")
    if enc_type == "transformer":
        return TransformerEncoder(**cfg)
    elif enc_type == "rnn":
        return RNNEncoder(**cfg)
    elif enc_type == "cnn":
        return CNNEncoder(**cfg)
    elif enc_type == "grouping":
        encoder = get_encoder_from_config(cfg.pop("encoder", {}))
        return GroupingEncoder(encoder, **cfg)
    else:
        raise ValueError(f"unknown encoder type {enc_type}")


def get_head_from_config(cfg: Dict[str, Any]) -> Head:
    head_type = cfg.pop("type")
    if head_type == "classification":
        return ClassificationHead(**cfg)
    elif head_type == "sequence_classification":
        return SequenceClassificationHead(**cfg)
    else:
        raise ValueError(f"unknown head type {head_type}")
