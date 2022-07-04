from typing import Dict, Optional, Tuple, Union

import tokenizers

import torch
from torch import nn

from whitespace_correction.model import tokenizer as tok_lib
from whitespace_correction.model.embedding import Embedding
from whitespace_correction.utils import common, constants, io, mask as mask_utils
from whitespace_correction.utils.config import EncoderDecoderConfig

logger = common.get_logger("DECODER")


class BaseDecoder(nn.Module):
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
        self.decoder: nn.Module
        self.out_proj_embedding: Union[nn.Identity, nn.Linear]
        self.out_proj: nn.Linear

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @property
    def decoder_model_dim(self) -> int:
        return self.config.model_dim


class PytorchDecoder(BaseDecoder):
    def __init__(self,
                 config: EncoderDecoderConfig,
                 device: torch.device,
                 custom_decoder_layer: nn.Module = None):
        super().__init__(config=config, device=device)

        self.tokenizer = tok_lib.load_tokenizer(self.config.tokenizer)
        self.padding_token_id = self.tokenizer.token_to_id(constants.PAD)

        self.embedding = Embedding(num_embeddings=self.tokenizer.get_vocab_size(),
                                   embedding_dim=self.config.embedding_dim,
                                   model_dim=self.config.model_dim,
                                   pad_token_id=self.tokenizer.token_to_id(constants.PAD),
                                   learned_positional_embeddings=self.config.learned_positional_embeddings,
                                   max_num_embeddings=self.config.max_num_embeddings,
                                   norm_embeddings=self.config.norm_embeddings,
                                   dropout=self.config.dropout)

        if custom_decoder_layer is not None:
            decoder_layer = custom_decoder_layer
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.model_dim,
                                                       nhead=self.config.attention_heads,
                                                       dim_feedforward=self.config.feedforward_dim,
                                                       dropout=self.config.dropout,
                                                       activation=self.config.activation)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                             num_layers=1 if self.config.share_parameters else self.config.num_layers)

        self.out_proj = nn.Linear(in_features=self.config.model_dim,
                                  out_features=self.tokenizer.get_vocab_size())

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"],
                               prefix="decoder.")
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        if self.config.fixed:
            for p in self.parameters():
                p.requires_grad = False

        self.to(device)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (tgt_mask is None or tgt_mask.dim() == 2), \
            f"tgt_mask has to be of shape [T, T], but got {tgt_mask.shape}"
        assert (memory_mask is None or memory_mask.dim() == 2), \
            f"memory_mask has to be of shape [T, S], but got {memory_mask.shape}"

        if tgt_mask is None:
            T, B = tgt.size()
            tgt_mask = mask_utils.generate_square_subsequent_mask(T, T, device=tgt.device)

        tgt_key_padding_mask = mask_utils.get_padding_mask(tgt, self.padding_token_id)

        emb = self.embedding(tgt)
        # reuse the same layer multiple times when parameters are shared
        if self.config.share_parameters:
            dec = emb
            for _ in range(self.config.num_layers):
                dec = self.decoder(dec, memory,
                                   tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask)
        else:
            dec = self.decoder(emb, memory,
                               tgt_mask=tgt_mask,
                               memory_mask=memory_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        return self.out_proj(dec), {}
