import copy
from typing import Dict, Any, Optional, List, Tuple

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules.embedding import Embedding, embedding_from_config
from text_correction_utils.modules.encoder import Encoder, encoder_from_config
from text_correction_utils.modules.head import Head, head_from_config


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
        emb, lengths, pos_emb = self.embedding(token_ids, lengths, **kwargs)
        enc = self.encoder(emb, lengths, pos_emb, **kwargs)
        output = self.head(enc, **kwargs)
        return output, self.encoder.additional_losses()


def model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")
    if model_type == "encoder_with_head":
        embedding = embedding_from_config(cfg["embedding"], input_tokenizer)
        encoder = encoder_from_config(cfg["encoder"])
        head = head_from_config(cfg["head"])
        return EncoderWithHead(embedding, encoder, head)

    else:
        raise ValueError(f"unknown model type {model_type}")
