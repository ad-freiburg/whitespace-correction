from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn.utils import rnn

from whitespace_correction.model.encoder import BaseEncoder
from whitespace_correction.model.heads import get_head_from_config
from whitespace_correction.model.mixins import EncoderMixin, InferenceMixin
from whitespace_correction.utils import common, inference, io, mask as mask_utils
from whitespace_correction.utils.config import RNNModelConfig, RNNEncoderDecoderConfig
from whitespace_correction.model import tokenizer as toklib, utils
from whitespace_correction.model.embedding import Embedding

logger = common.get_logger("RNN")


class PytorchEncoder(BaseEncoder):
    def __init__(self,
                 config: RNNEncoderDecoderConfig,
                 device: torch.device):
        super().__init__(config=config, device=device)
        self.config: RNNEncoderDecoderConfig

        self.tokenizer = toklib.get_tokenizer_from_config(self.config.tokenizer)

        self.embedding = Embedding(num_embeddings=self.tokenizer.vocab_size,
                                   embedding_dim=self.config.embedding_dim,
                                   model_dim=self.config.model_dim,
                                   pad_token_id=self.tokenizer.pad_token_id,
                                   norm_embeddings=self.config.norm_embeddings,
                                   dropout=self.config.dropout)

        assert self.config.type in {"lstm", "gru"}, \
            f"rnn type must be one of lstm or gru, but got {self.config.type}"
        if self.config.type == "lstm":
            self.encoder = nn.LSTM(
                self.config.model_dim,
                self.config.model_dim,
                num_layers=self.config.num_layers,
                bidirectional=self.config.bidirectional,
                dropout=self.config.dropout * (self.config.num_layers > 1)
            )
        else:
            self.encoder = nn.GRU(
                self.config.model_dim,
                self.config.model_dim,
                num_layers=self.config.num_layers,
                bidirectional=self.config.bidirectional,
                dropout=self.config.dropout * (self.config.num_layers > 1)
            )

        self.agg_fn = utils.get_aggregation_fn(self.config.group_aggregation)
        self.group_name = self.config.group_name
        self.group_at = self.config.group_at

        self.padding_token_id = self.tokenizer.pad_token_id

    def forward(self,
                src: torch.Tensor,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        padding_mask = mask_utils.get_padding_mask_from_token_ids(src, self.padding_token_id)
        emb = self.embedding(src)

        if self.group_name and self.group_at == "before":
            emb, src_lengths = utils.group_features(emb, kwargs[self.group_name], self.agg_fn)
            src_lengths = torch.tensor(src_lengths, dtype=torch.long)
        else:
            src_lengths = torch.logical_not(padding_mask).sum(1).long().cpu()

        packed = rnn.pack_padded_sequence(emb, src_lengths, enforce_sorted=False)
        packed, _ = self.encoder(packed)
        unpacked = rnn.unpack_sequence(packed)
        padded = rnn.pad_sequence(unpacked)
        if self.config.bidirectional:
            # average forward and backward features when using a bidirectional rnn
            padded = einops.rearrange(padded, "s b (d h) -> s b d h", d=2).mean(2)

        if self.group_name and self.group_at == "after":
            padded, _ = utils.group_features(padded, kwargs[self.group_name], self.agg_fn)

        return padded, {}


class RNNEncoderModelWithHead(nn.Module, EncoderMixin, InferenceMixin):
    def __init__(self,
                 config: RNNModelConfig,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        assert self.config.encoder is not None, \
            "encoder config must be specified when using rnn_encoder_with_head model"
        self.encoder = PytorchEncoder(config=self.config.encoder, device=self.device)

        assert self.config.head is not None, "head config must be specified when using rnn_encoder_with_head model"
        self.head = get_head_from_config(config=self.config.head,
                                         model_dim=self.config.encoder.model_dim)

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"])
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        self.to(self.device)

    def encode(self,
               src: torch.Tensor,
               **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.encoder(src=src, **kwargs)

    def forward(self,
                input_ids: torch.Tensor,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        encodings, enc_loss_dict = self.encode(input_ids, **kwargs)
        return self.head(encodings, **kwargs), enc_loss_dict

    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> List[inference.InferenceResult]:
        assert not self.training, "model cannot be in training mode during inference"

        input_ids: torch.Tensor = inference.sequences_to_ids(sequences=sequences,
                                                             tokenizer=self.encoder.tokenizer,
                                                             device=self.device)

        encoder_outputs, encoder_padding_mask = inference.batch_encode(
            model=self,
            input_ids=input_ids,
            max_length=self.encoder.config.max_num_embeddings,
            device=self.device,
            return_padding_mask=True
        )

        kwargs["input_lengths"] = torch.sum(torch.logical_not(encoder_padding_mask), dim=1)
        return self.head.inference(
            encodings=encoder_outputs,
            **kwargs
        )
