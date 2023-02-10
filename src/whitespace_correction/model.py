import copy
from typing import Dict, Any, Optional, Tuple, Union

import torch
from torch import nn

from text_correction_utils import tokenization
from text_correction_utils.modules.embedding import Embedding, embedding_from_config
from text_correction_utils.modules.encoder import Encoder, encoder_from_config
from text_correction_utils.modules.decoder import Decoder, decoder_from_config
from text_correction_utils.modules.head import Head, head_from_config

from transformers import T5EncoderModel, T5Config


class Model(nn.Module):
    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class PretrainedEncoderWithHead(Model):
    def __init__(
        self,
        encoder: Encoder,
        head: Head
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc, kwargs = self.encoder(token_ids, **kwargs)
        output = self.head(enc, **kwargs)
        return output, self.encoder.additional_losses()


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
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        emb, pos_emb = self.embedding(token_ids, **kwargs)
        enc, kwargs = self.encoder(emb, pos=pos_emb, **kwargs)
        output = self.head(enc, **kwargs)
        return output, self.encoder.additional_losses()


class EncoderDecoderWithHead(Model):
    def __init__(
        self,
        encoder_embedding: Embedding,
        encoder: Encoder,
        decoder: Decoder,
        head: Union[Head, Tuple[int, int]],
        share_encoder_embedding_with_decoder_embedding: bool = False,
        share_decoder_embedding_with_head: bool = False,
        decoder_embedding: Optional[Embedding] = None,
    ):
        super().__init__()
        self.encoder_embedding = encoder_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

        if share_encoder_embedding_with_decoder_embedding:
            self.decoder_embedding = self.encoder_embedding
        else:
            assert decoder_embedding is not None, \
                "decoder_embedding must be provided if not sharing encoder embedding with decoder embedding"
            self.decoder_embedding = decoder_embedding

        self.share_decoder_embedding_with_head = share_decoder_embedding_with_head
        if share_decoder_embedding_with_head:
            assert isinstance(head, tuple) and len(head) == 2 and all(isinstance(x, int) for x in head), \
                "head must be a tuple of two ints specifying model dim and number of outputs if sharing decoder embedding with head"
            dim, outputs = head
            self.head = nn.Linear(dim, outputs)
            self.head.weight = self.decoder_embedding.embedding.emb.weight
        else:
            assert head is not None, \
                "head must be provided if not sharing decoder embedding with head"
            self.head = head

    def encode(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        emb, pos_emb = self.encoder_embedding(token_ids, **kwargs)
        enc, kwargs = self.encoder(emb, pos_emb, **kwargs)
        return enc, kwargs

    def decode(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        emb, pos_emb = self.decoder_embedding(token_ids, **kwargs)
        dec, kwargs = self.decoder(
            emb,
            pos_emb,
            **kwargs
        )
        if self.share_decoder_embedding_with_head:
            return self.head(dec)
        else:
            return self.head(dec, **kwargs)

    def forward(
        self,
        _: torch.Tensor,
        **__: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class ByT5Encoder(Encoder):
    def __init__(
        self,
        name: str,
        pad_token_id: int,
        pretrained: bool = True
    ):
        super().__init__()
        if pretrained:
            self.encoder = T5EncoderModel.from_pretrained(name)
        else:
            config = T5Config.from_pretrained(name)
            self.encoder = T5EncoderModel(config)

        self.pad_token_id = pad_token_id

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc = self.encoder(
            input_ids=token_ids,
            attention_mask=(token_ids != self.pad_token_id).long()
        ).last_hidden_state
        return enc, kwargs


class ByT5WithHead(Model):
    def __init__(
        self,
        encoder: ByT5Encoder,
        head: Head,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        token_ids: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        enc, kwargs = self.encoder(token_ids, **kwargs)
        return self.head(enc, **kwargs)


def pretrained_encoder_from_config(cfg: Dict[str, Any], input_tokenizer: tokenization.Tokenizer) -> Encoder:
    if cfg["type"] == "byt5":
        pad_token_id = input_tokenizer.pad_token_id()
        assert "name" in cfg, "name must be provided for byt5 encoder"
        return ByT5Encoder(cfg["name"], pad_token_id)
    else:
        raise ValueError(f"unknown pretrained encoder type {cfg['type']}")


def model_from_config(
    cfg: Dict[str, Any],
    input_tokenizer: tokenization.Tokenizer,
    output_tokenizer: Optional[tokenization.Tokenizer],
) -> Model:
    cfg = copy.deepcopy(cfg)
    model_type = cfg.pop("type")
    if model_type == "pretrained_encoder_with_head":
        encoder = encoder_from_config(
            cfg["encoder"],
            additional_encoder_fn=lambda cfg: pretrained_encoder_from_config(cfg, input_tokenizer)
        )
        head = head_from_config(cfg["head"])
        return PretrainedEncoderWithHead(encoder, head)
    elif model_type == "encoder_with_head":
        embedding = embedding_from_config(cfg["embedding"], input_tokenizer)
        encoder = encoder_from_config(cfg["encoder"])
        head = head_from_config(cfg["head"])
        return EncoderWithHead(embedding, encoder, head)
    elif model_type == "encoder_decoder_with_head":
        encoder_embedding = embedding_from_config(cfg["encoder_embedding"], input_tokenizer)
        encoder = encoder_from_config(cfg["encoder"])
        decoder = decoder_from_config(cfg["decoder"])
        share_encoder_embedding_with_decoder_embedding = cfg.get(
            "share_encoder_embedding_with_decoder_embedding", False
        )
        if share_encoder_embedding_with_decoder_embedding:
            decoder_embedding = None
        else:
            assert "decoder_embedding" in cfg, \
                "decoder_embedding must be in config if not sharing encoder embedding with decoder embedding"
            decoder_embedding = embedding_from_config(cfg["decoder_embedding"], output_tokenizer)
        share_decoder_embedding_with_head = cfg.get("share_decoder_embedding_with_head", False)
        if share_decoder_embedding_with_head:
            head = (cfg["decoder"]["dim"], output_tokenizer.vocab_size())
        else:
            head = head_from_config(cfg["head"])
        return EncoderDecoderWithHead(
            encoder_embedding,
            encoder,
            decoder,
            head,
            share_encoder_embedding_with_decoder_embedding,
            share_decoder_embedding_with_head,
            decoder_embedding
        )
    else:
        raise ValueError(f"unknown model type {model_type}")
