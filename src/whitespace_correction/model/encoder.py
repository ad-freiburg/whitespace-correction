from typing import Dict, Tuple, Any

import torch
from torch import nn

from whitespace_correction.model.tokenizer import Tokenizer
from whitespace_correction.model.embedding import Embedding
from whitespace_correction.utils.config import EncoderDecoderConfig


class BaseEncoder(nn.Module):
    def __init__(self,
                 config: EncoderDecoderConfig,
                 device: torch.device):
        super().__init__()

        self.config = config
        self.device = device

        # set these attributes in child class
        self.tokenizer: Tokenizer
        self.padding_token_id: int
        self.embedding: Embedding
        self.encoder: nn.Module

    def forward(self,
                src: torch.Tensor,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()
