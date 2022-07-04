from typing import Dict, Optional, Tuple

import tokenizers

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn

from whitespace_correction.model import tokenizer as toklib
from whitespace_correction.model.embedding import Embedding
from whitespace_correction.utils import common, constants, io, mask as mask_utils
from whitespace_correction.utils.config import EncoderDecoderConfig

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


# exact copy of pytorch native transformer encoder layer, just with need_weights set to true and approximate gelu
class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            **factory_kwargs
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


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
            encoder_layer = TransformerEncoderLayer(
                d_model=self.config.model_dim,
                nhead=self.config.attention_heads,
                dim_feedforward=self.config.feedforward_dim,
                dropout=self.config.dropout,
                activation=self.config.activation
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=1 if self.config.share_parameters else self.config.num_layers)

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"],
                               prefix="decoder." if as_decoder else "encoder.")
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        if self.config.fixed:
            for p in self.parameters():
                p.requires_grad = False

        self.to(self.device)

        # for torchscript compatibility: set config values directly
        self.share_parameters = self.config.share_parameters
        self.num_layers = self.config.num_layers

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
        if self.share_parameters:
            enc = emb
            for _ in range(self.num_layers):
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
