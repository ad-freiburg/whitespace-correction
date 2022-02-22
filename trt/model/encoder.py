from typing import Dict, Optional, Tuple

import tokenizers
import torch
from torch import nn

from trt.model import tokenizer as toklib
from trt.model.embedding import Embedding
from trt.utils import common, constants, io, mask as mask_utils
from trt.utils.config import EncoderDecoderConfig

logger = common.get_logger("ENCODER")


class BaseEncoder(nn.Module):
    def __init__(self,
                 config: EncoderDecoderConfig,
                 device: torch.device):
        super().__init__()

        self.config = config
        self.device = device

        # set these attributes in child class
        self.tokenizer: tokenizers.Tokenizer
        self.padding_token_id: int
        self.embedding: Embedding
        self.encoder: nn.Module

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @property
    def encoder_model_dim(self) -> int:
        return self.config.model_dim


class PytorchEncoder(BaseEncoder):
    def __init__(self,
                 config: EncoderDecoderConfig,
                 device: torch.device,
                 custom_encoder_layer: nn.Module = None,
                 as_decoder: bool = False):
        super().__init__(config=config, device=device)
        self.as_decoder = as_decoder

        self.tokenizer = toklib.load_tokenizer(self.config.tokenizer)
        self.padding_token_id = self.tokenizer.token_to_id(constants.PAD)

        self.embedding = Embedding(num_embeddings=self.tokenizer.get_vocab_size(),
                                   embedding_dim=self.config.embedding_dim,
                                   model_dim=self.config.model_dim,
                                   pad_token_id=self.tokenizer.token_to_id(constants.PAD),
                                   learned_positional_embeddings=self.config.learned_positional_embeddings,
                                   max_num_embeddings=self.config.max_num_embeddings,
                                   norm_embeddings=self.config.norm_embeddings,
                                   dropout=self.config.dropout)

        if custom_encoder_layer is not None:
            encoder_layer = custom_encoder_layer
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.model_dim,
                                                       nhead=self.config.attention_heads,
                                                       dim_feedforward=self.config.feedforward_dim,
                                                       dropout=self.config.dropout,
                                                       activation=self.config.activation)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=1 if self.config.share_parameters else self.config.num_layers)

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"],
                               match_suffixes=True)
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        if self.config.fixed:
            for p in self.parameters():
                p.requires_grad = False

        self.to(self.device)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (src_mask is None or src_mask.dim() == 2), \
            f"src_mask has to be of shape [S, S], but got {src_mask.shape}"

        if self.as_decoder and src_mask is None:
            S, B = src.shape
            src_mask = mask_utils.generate_square_subsequent_mask(S, S, device=src.device)

        src_key_padding_mask = mask_utils.get_padding_mask(src, self.padding_token_id)

        emb = self.embedding(src)
        # reuse the same layer multiple time when parameters are shared
        if self.config.share_parameters:
            enc = emb
            for _ in range(self.config.num_layers):
                enc = self.encoder(enc, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            enc = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return enc, {}


def get_encoder_from_config(config: EncoderDecoderConfig, device: torch.device) -> BaseEncoder:
    if config.type == "default":
        encoder = PytorchEncoder(config=config,
                                 device=device)

    else:
        raise ValueError(f"Unknown encoder type {config.type}")

    return encoder
