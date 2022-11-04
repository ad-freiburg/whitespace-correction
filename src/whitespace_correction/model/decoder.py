from typing import Dict, Tuple, Union, Any

import torch
from torch import nn

from whitespace_correction.model.tokenizer import Tokenizer
from whitespace_correction.model.embedding import Embedding
from whitespace_correction.utils.config import EncoderDecoderConfig


class BaseDecoder(nn.Module):
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
        self.decoder: nn.Module
        self.out_proj_embedding: Union[nn.Identity, nn.Linear]
        self.out_proj: nn.Linear

    def forward(self,
                tgt: torch.Tensor,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()
