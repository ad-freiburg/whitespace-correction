from typing import Any, Dict, List, Optional, Tuple, Union

import tokenizers

import torch
from torch import nn

from whitespace_correction.model.decoder import BaseDecoder, PytorchDecoder
from whitespace_correction.model.encoder import BaseEncoder, PytorchEncoder, get_encoder_from_config
from whitespace_correction.model.heads import get_head_from_config
from whitespace_correction.model.mixins import DecoderMixin, EncoderMixin, InferenceMixin
from whitespace_correction.utils import common, constants, inference, io, mask as mask_utils
from whitespace_correction.utils.config import ModelConfig

logger = common.get_logger("MODEL")


class TransformerModel(nn.Module, EncoderMixin, DecoderMixin, InferenceMixin):
    def __init__(self,
                 config: ModelConfig,
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
            self.encoder = get_encoder_from_config(config=self.config.encoder,
                                                   device=self.device)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            assert self.config.decoder is not None, "decoder config must be specified when using transformer model"
            self.decoder = PytorchDecoder(config=self.config.decoder,
                                          device=device)

        if self.config.share_encoder_decoder_embeddings:
            assert (self.encoder.config.embedding_dim == self.decoder.config.embedding_dim
                    and self.encoder.config.learned_positional_embeddings ==
                    self.decoder.config.learned_positional_embeddings
                    and self.encoder.config.max_num_embeddings == self.decoder.config.max_num_embeddings
                    and self.encoder.tokenizer.get_vocab_size() == self.decoder.tokenizer.get_vocab_size()
                    and self.encoder.config.dropout == self.decoder.config.dropout), \
                f"to share the embeddings between the encoder and decoder, they must have the same " \
                f"embedding dimensionality " \
                f"(got {self.encoder.config.embedding_dim} and {self.decoder.config.embedding_dim}), " \
                f"maximum number of embeddings " \
                f"(got {self.encoder.config.max_num_embeddings} and {self.decoder.config.max_num_embeddings}), " \
                f"vocab size " \
                f"(got {self.encoder.tokenizer.get_vocab_size()} and {self.decoder.tokenizer.get_vocab_size()}) " \
                f"and dropout " \
                f"(got {self.encoder.config.dropout} and {self.decoder.config.dropout}) " \
                f"and have the same setting for learned positional embeddings " \
                f"(got {self.encoder.config.learned_positional_embeddings} " \
                f"and {self.decoder.config.learned_positional_embeddings})"
            self.decoder.embedding = self.encoder.embedding

        if self.config.share_decoder_input_output_embeddings:
            assert self.decoder.config.embedding_dim == self.decoder.config.model_dim, \
                "to share decoder input and output embeddings, the embedding and model dim must be the same"

            self.decoder.out_proj.weight = self.decoder.embedding.embedding.weight

        if self.encoder.config.model_dim != self.decoder.config.model_dim:
            self.cross_proj = nn.Linear(self.encoder.encoder_model_dim, self.decoder.decoder_model_dim)
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

    def get_memory_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return mask_utils.get_padding_mask(src, self.encoder.padding_token_id)

    def encode(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        outputs = self.encoder(src=src, src_mask=src_mask)
        return outputs

    def decode(self,
               tgt: torch.Tensor,
               memory: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory = self.cross_proj(memory)

        outputs = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
        return outputs

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        memory, enc_loss_dict = self.encode(src=src, src_mask=src_mask)

        memory_key_padding_mask = self.get_memory_key_padding_mask(src)

        output, dec_loss_dict = self.decode(tgt=tgt,
                                            memory=memory,
                                            tgt_mask=tgt_mask,
                                            memory_mask=memory_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
        enc_loss_dict.update(dec_loss_dict)

        return output, enc_loss_dict

    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> Union[List[List[inference.InferenceResult]], List[inference.InferenceResult]]:
        assert not self.training, "model cannot be in training mode during inference"

        input_ids = inference.sequences_to_ids(sequences=sequences,
                                               tokenizer=self.encoder.tokenizer,
                                               device=self.device)

        return inference.inference_with_ids(model=self,
                                            input_ids=input_ids,
                                            bos_token_id=self.decoder.tokenizer.token_to_id(constants.BOS),
                                            eos_token_id=self.decoder.tokenizer.token_to_id(constants.EOS),
                                            max_input_length=kwargs.get(
                                                "max_input_length", self.encoder.config.max_num_embeddings
                                            ),
                                            max_output_length=kwargs.get(
                                                "max_output_length", self.decoder.config.max_num_embeddings
                                            ),
                                            device=self.device,
                                            method=kwargs.pop("inference_method", "greedy"),
                                            **kwargs)


class TransformerEncoderModelWithHead(nn.Module, EncoderMixin, InferenceMixin):
    def __init__(self,
                 config: ModelConfig,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        assert self.config.encoder is not None, "encoder config must be specified when using encoder_with_head model"
        self.encoder = get_encoder_from_config(config=self.config.encoder,
                                               device=self.device)

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
               src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.encoder(src=src, src_mask=src_mask)

    def get_memory_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return mask_utils.get_padding_mask(src, self.encoder.padding_token_id)

    def forward(self, input_ids: torch.Tensor, src_mask: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                                                       Dict[str, torch.Tensor]]:
        encodings, enc_loss_dict = self.encode(input_ids, src_mask)
        return self.head(encodings), enc_loss_dict

    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> List[inference.InferenceResult]:
        assert not self.training, "model cannot be in training mode during inference"

        input_ids: torch.Tensor = inference.sequences_to_ids(sequences=sequences,
                                                             tokenizer=self.encoder.tokenizer,
                                                             device=self.device)

        encoder_outputs = inference.batch_encode(model=self,
                                                 input_ids=input_ids,
                                                 max_length=self.encoder.config.max_num_embeddings,
                                                 device=self.device)

        input_lengths = (input_ids.cpu() != self.encoder.tokenizer.token_to_id(constants.PAD)).sum(1)
        return self.head.inference(encodings=encoder_outputs,
                                   input_lengths=input_lengths,
                                   **kwargs)


class TransformerDecoderModel(nn.Module, DecoderMixin, InferenceMixin):
    def __init__(self,
                 config: ModelConfig,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        # for decoder only models we use an encoder with masked self-attention since cross-attention is not used
        assert self.config.decoder is not None, "decoder config must be specified when using decoder model"
        self.decoder = PytorchEncoder(config=self.config.decoder,
                                      device=self.device,
                                      as_decoder=True)

        self.out_proj = nn.Linear(in_features=self.config.decoder.model_dim,
                                  out_features=self.decoder.tokenizer.get_vocab_size())

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

        decoder_outputs, dec_loss_dict = self.decoder(src=tgt, src_mask=tgt_mask)
        return self.out_proj(decoder_outputs), dec_loss_dict

    def forward(self, input_ids: torch.Tensor, src_mask: torch.Tensor = None) -> Tuple[torch.Tensor,
                                                                                       Dict[str, torch.Tensor]]:
        return self.decode(input_ids, src_mask)

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

        return inference.inference_with_ids(model=self,
                                            bos_token_id=self.decoder.tokenizer.token_to_id(constants.BOS),
                                            eos_token_id=self.decoder.tokenizer.token_to_id(constants.EOS),
                                            max_length=kwargs.pop("max_length", self.decoder.config.max_num_embeddings),
                                            device=self.device,
                                            method=kwargs.pop("inference_method", "greedy"),
                                            input_ids=None,
                                            decoder_only=True,
                                            decoder_input_ids=input_ids,
                                            decoder_padding_token_id=self.decoder.padding_token_id,
                                            **kwargs)


def get_model_from_config(config: ModelConfig,
                          device: torch.device) -> Union[TransformerEncoderModelWithHead,
                                                         TransformerDecoderModel,
                                                         TransformerModel]:
    if config.type == "encoder_with_head":
        assert config.encoder is not None and config.head is not None
        model = TransformerEncoderModelWithHead(config=config,
                                                device=device)

    elif config.type == "decoder":
        assert config.decoder is not None
        model = TransformerDecoderModel(config=config,
                                        device=device)

    elif config.type == "transformer":
        assert config.encoder is not None and config.decoder is not None
        model = TransformerModel(config=config,
                                 device=device)
    else:
        raise ValueError(f"Unknown model type {config.type}")

    return model


def get_tokenizers_from_model(model: nn.Module) -> Tuple[tokenizers.Tokenizer,
                                                         tokenizers.Tokenizer]:
    if hasattr(model, "encoder"):
        encoder_tokenizer = model.encoder.tokenizer
    else:
        encoder_tokenizer = None

    if hasattr(model, "decoder"):
        decoder_tokenizer = model.decoder.tokenizer
    else:
        decoder_tokenizer = None

    assert encoder_tokenizer is not None or decoder_tokenizer is not None, \
        "Could neither find encoder or decoder tokenizer"

    encoder_tokenizer = encoder_tokenizer if encoder_tokenizer is not None else decoder_tokenizer
    decoder_tokenizer = decoder_tokenizer if decoder_tokenizer is not None else encoder_tokenizer

    return encoder_tokenizer, decoder_tokenizer
