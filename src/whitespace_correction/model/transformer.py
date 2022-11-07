from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn

from whitespace_correction.model.decoder import BaseDecoder
from whitespace_correction.model.encoder import BaseEncoder
from whitespace_correction.model.heads import get_head_from_config
from whitespace_correction.model.mixins import DecoderMixin, EncoderMixin, InferenceMixin
from whitespace_correction.utils import common, inference, io, mask as mask_utils
from whitespace_correction.utils.config import TransformerModelConfig, TransformerEncoderDecoderConfig
from whitespace_correction.model import tokenizer as toklib, utils
from whitespace_correction.model.embedding import Embedding

logger = common.get_logger("TRANSFORMER")


# exact copy of pytorch native transformer encoder layer, just with need_weights set to true
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
                 config: TransformerEncoderDecoderConfig,
                 device: torch.device,
                 custom_encoder_layer: nn.Module = None):
        super().__init__(config=config, device=device)

        self.tokenizer = toklib.get_tokenizer_from_config(self.config.tokenizer)

        self.embedding = Embedding(num_embeddings=self.tokenizer.vocab_size,
                                   embedding_dim=self.config.embedding_dim,
                                   model_dim=self.config.model_dim,
                                   pad_token_id=self.tokenizer.pad_token_id,
                                   positional_embeddings=self.config.positional_embeddings,
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
                               prefix="encoder.")
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        if self.config.fixed:
            for p in self.parameters():
                p.requires_grad = False

        self.agg_fn = utils.get_aggregation_fn(self.config.group_aggregation)

        self.to(self.device)

        # for torchscript compatibility: set config values directly
        self.share_parameters = self.config.share_parameters
        self.num_layers = self.config.num_layers
        self.group_name = self.config.group_name
        self.group_at = self.config.group_at
        self.padding_token_id = self.tokenizer.pad_token_id

    def forward(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (src_mask is None or src_mask.dim() == 2), \
            f"src_mask has to be of shape [S, S], but got {src_mask.shape}"

        emb = self.embedding(src)

        if self.group_name and self.group_at == "before":
            emb, lengths = utils.group_features(emb, kwargs[self.group_name], self.agg_fn)
            src_key_padding_mask = mask_utils.get_padding_mask_from_lengths(lengths, emb.device)
        else:
            src_key_padding_mask = mask_utils.get_padding_mask_from_token_ids(src, self.padding_token_id)

        # reuse the same layer multiple time when parameters are shared
        if self.share_parameters:
            enc = emb
            for _ in range(self.num_layers):
                enc = self.encoder(enc, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            enc = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        if self.group_name and self.group_at == "after":
            enc, _ = utils.group_features(enc, kwargs[self.group_name], self.agg_fn)

        return enc, {}


class PytorchDecoder(BaseDecoder):
    def __init__(self,
                 config: TransformerEncoderDecoderConfig,
                 device: torch.device,
                 custom_decoder_layer: nn.Module = None,
                 decoder_only: bool = False):
        super().__init__(config=config, device=device)
        self.decoder_only = decoder_only

        self.tokenizer = toklib.get_tokenizer_from_config(self.config.tokenizer)
        self.padding_token_id = self.tokenizer.pad_token_id

        self.embedding = Embedding(num_embeddings=self.tokenizer.vocab_size,
                                   embedding_dim=self.config.embedding_dim,
                                   model_dim=self.config.model_dim,
                                   pad_token_id=self.tokenizer.pad_token_id,
                                   positional_embeddings=self.config.positional_embeddings,
                                   max_num_embeddings=self.config.max_num_embeddings,
                                   norm_embeddings=self.config.norm_embeddings,
                                   dropout=self.config.dropout)

        if self.decoder_only:
            decoder_layer = TransformerEncoderLayer(d_model=self.config.model_dim,
                                                    nhead=self.config.attention_heads,
                                                    dim_feedforward=self.config.feedforward_dim,
                                                    dropout=self.config.dropout,
                                                    activation=self.config.activation)
            self.decoder = nn.TransformerEncoder(
                encoder_layer=decoder_layer,
                num_layers=1 if self.config.share_parameters else self.config.num_layers
            )
        else:
            if custom_decoder_layer is not None:
                decoder_layer = custom_decoder_layer
            else:
                decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.model_dim,
                                                           nhead=self.config.attention_heads,
                                                           dim_feedforward=self.config.feedforward_dim,
                                                           dropout=self.config.dropout,
                                                           activation=self.config.activation)
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=1 if self.config.share_parameters else self.config.num_layers
            )

        self.out_proj = nn.Linear(in_features=self.config.model_dim,
                                  out_features=self.tokenizer.vocab_size)

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
                memory: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert (tgt_mask is None or tgt_mask.dim() == 2), \
            f"tgt_mask has to be of shape [T, T], but got {tgt_mask.shape}"
        assert (memory_mask is None or memory_mask.dim() == 2), \
            f"memory_mask has to be of shape [T, S], but got {memory_mask.shape}"

        if tgt_mask is None:
            T, B = tgt.shape
            tgt_mask = mask_utils.generate_square_subsequent_mask(T, T, device=tgt.device)

        tgt_key_padding_mask = mask_utils.get_padding_mask_from_token_ids(tgt, self.padding_token_id)

        # we need to pass different kwargs to the underlying decoder module when we use a decoder only decoder
        # because we implement it as an encoder under the hood
        if self.decoder_only:
            kwargs.update({
                "mask": tgt_mask,
                "src_key_padding_mask": tgt_key_padding_mask
            })
        else:
            kwargs.update({
                "memory": memory,
                "tgt_mask": tgt_mask,
                "memory_mask": memory_mask,
                "tgt_key_padding_mask": tgt_key_padding_mask,
                "memory_key_padding_mask": memory_key_padding_mask
            })

        emb = self.embedding(tgt)
        # reuse the same layer multiple times when parameters are shared
        if self.config.share_parameters:
            dec = emb
            for _ in range(self.config.num_layers):
                dec = self.decoder(dec, **kwargs)
        else:
            dec = self.decoder(emb, **kwargs)
        return self.out_proj(dec), {}


class TransformerModel(nn.Module, EncoderMixin, DecoderMixin, InferenceMixin):
    def __init__(self,
                 config: TransformerModelConfig,
                 device: torch.device,
                 custom_encoder: BaseEncoder = None,
                 custom_decoder: BaseDecoder = None):
        super().__init__()
        self.config = config
        self.device = device

        if custom_encoder is not None:
            self.encoder = custom_encoder

        else:
            assert self.config.encoder is not None, "encoder config must be specified when using transformer model"
            self.encoder = PytorchEncoder(config=self.config.encoder, device=self.device)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            assert self.config.decoder is not None, "decoder config must be specified when using transformer model"
            self.decoder = PytorchDecoder(config=self.config.decoder,
                                          device=device)

        if self.config.share_encoder_decoder_embeddings:
            assert (
                    self.encoder.config.embedding_dim == self.decoder.config.embedding_dim
                    and self.encoder.config.positional_embeddings == self.decoder.config.positional_embeddings
                    and self.encoder.config.max_num_embeddings == self.decoder.config.max_num_embeddings
                    and self.encoder.tokenizer.vocab_size == self.decoder.tokenizer.vocab_size
                    and self.encoder.config.dropout == self.decoder.config.dropout
            ), \
                f"to share the embeddings between the encoder and decoder, they must have the same " \
                f"embedding dimensionality " \
                f"(got {self.encoder.config.embedding_dim} and {self.decoder.config.embedding_dim}), " \
                f"maximum number of embeddings " \
                f"(got {self.encoder.config.max_num_embeddings} and {self.decoder.config.max_num_embeddings}), " \
                f"vocab size " \
                f"(got {self.encoder.tokenizer.get_vocab_size()} and {self.decoder.tokenizer.get_vocab_size()}) " \
                f"and dropout " \
                f"(got {self.encoder.config.dropout} and {self.decoder.config.dropout}) " \
                f"and have the same setting for positional embeddings " \
                f"(got {self.encoder.config.positional_embeddings} " \
                f"and {self.decoder.config.positional_embeddings})"
            self.decoder.embedding = self.encoder.embedding

        if self.config.share_decoder_input_output_embeddings:
            assert self.decoder.config.embedding_dim == self.decoder.config.model_dim, \
                "to share decoder input and output embeddings, the embedding and model dim must be the same"

            self.decoder.out_proj.weight = self.decoder.embedding.embedding.weight

        if self.encoder.config.model_dim != self.decoder.config.model_dim:
            self.cross_proj = nn.Linear(self.encoder.config.model_dim, self.decoder.config.model_dim)
        else:
            self.cross_proj = nn.Identity()

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"])
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        self.to(device)

    def encode(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.encoder(src=src, src_mask=src_mask, **kwargs)
        return outputs

    def decode(self,
               tgt: torch.Tensor,
               memory: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None,
               **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory = self.cross_proj(memory)

        outputs = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask, **kwargs
        )
        return outputs

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, enc_loss_dict = self.encode(src=src, src_mask=src_mask, **kwargs)

        memory_key_padding_mask = self.get_encoder_padding_mask(src)

        output, dec_loss_dict = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            **kwargs
        )
        enc_loss_dict.update(dec_loss_dict)

        return output, enc_loss_dict

    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> Union[List[List[inference.InferenceResult]], List[inference.InferenceResult]]:
        assert not self.training, "model cannot be in training mode during inference"

        input_ids = inference.sequences_to_ids(sequences=sequences,
                                               tokenizer=self.encoder.tokenizer,
                                               device=self.device)

        return inference.inference_with_ids(
            model=self,
            input_ids=input_ids,
            bos_token_id=self.decoder.tokenizer.bos_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            max_input_length=kwargs.get("max_input_length", self.encoder.config.max_num_embeddings),
            max_output_length=kwargs.get("max_output_length", self.decoder.config.max_num_embeddings),
            device=self.device,
            method=kwargs.pop("inference_method", "greedy"),
            **kwargs
        )


class TransformerEncoderModelWithHead(nn.Module, EncoderMixin, InferenceMixin):
    def __init__(self,
                 config: TransformerModelConfig,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        assert self.config.encoder is not None, "encoder config must be specified when using encoder_with_head model"
        self.encoder = PytorchEncoder(config=self.config.encoder, device=self.device)

        assert self.config.head is not None, "head config must be specified when using encoder_with_head model"
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

    def inference(
            self,
            sequences: Union[str, List[str]],
            **kwargs: Any
    ) -> List[inference.InferenceResult]:
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


class TransformerDecoderModel(nn.Module, DecoderMixin, InferenceMixin):
    def __init__(self,
                 config: TransformerModelConfig,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # for decoder only models we use a decoder without cross-attention (basically an encoder with masking)
        assert self.config.decoder is not None, "decoder config must be specified when using decoder model"
        self.decoder = PytorchDecoder(config=self.config.decoder,
                                      device=self.device,
                                      decoder_only=True)

        self.out_proj = nn.Linear(in_features=self.config.decoder.model_dim,
                                  out_features=self.decoder.tokenizer.vocab_size)

        if self.config.pretrained:
            checkpoint = io.load_checkpoint(self.config.pretrained)
            io.load_state_dict(module=self,
                               state_dict=checkpoint["model_state_dict"])
            global logger
            logger.info(f"Successfully loaded pretrained weights into {self.__class__.__name__} "
                        f"from {self.config.pretrained}")

        if self.config.share_decoder_input_output_embeddings:
            assert self.decoder.config.embedding_dim == self.decoder.config.model_dim, \
                "to share decoder input and output embeddings, the embedding and model dim must be the same"

            self.out_proj.weight = self.decoder.embedding.embedding.weight

        self.to(self.device)

    def decode(self,
               tgt: torch.Tensor,
               memory: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert memory is None, "in decoder only models memory is not used and should be None"
        assert memory_mask is None, "in decoder only models memory mask is not used and should be None"
        assert memory_key_padding_mask is None, "in decoder only models memory key padding mask is " \
                                                "not used and should be None"

        decoder_outputs, dec_loss_dict = self.decoder(tgt=tgt, tgt_mask=tgt_mask)
        return self.out_proj(decoder_outputs), dec_loss_dict

    def forward(self, input_ids: torch.Tensor, src_mask: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                                                       Dict[str, torch.Tensor]]:
        return self.decode(tgt=input_ids, tgt_mask=src_mask)

    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> Union[List[List[inference.InferenceResult]], List[inference.InferenceResult]]:
        assert not self.training, "model cannot be in training mode during inference"

        input_ids = inference.sequences_to_ids(sequences=sequences,
                                               tokenizer=self.decoder.tokenizer,
                                               device=self.device,
                                               as_list=True)
        # remove eos tokens
        input_ids = [token_ids[:-1] for token_ids in input_ids]

        return inference.inference_with_ids(
            model=self,
            bos_token_id=self.decoder.tokenizer.bos_token_id,
            eos_token_id=self.decoder.tokenizer.eos_token_id,
            max_length=kwargs.pop("max_length", self.decoder.config.max_num_embeddings),
            device=self.device,
            method=kwargs.pop("inference_method", "greedy"),
            input_ids=None,
            decoder_only=True,
            decoder_input_ids=input_ids,
            decoder_padding_token_id=self.decoder.padding_token_id,
            **kwargs
        )
