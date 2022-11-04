from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from whitespace_correction.model.decoder import BaseDecoder
from whitespace_correction.model.encoder import BaseEncoder
from whitespace_correction.utils import mask as mask_utils, constants


class EncoderMixin:
    # set encoder in child class
    encoder: BaseEncoder

    def encode(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def get_encoder_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return mask_utils.get_padding_mask_from_token_ids(src, self.encoder.padding_token_id)

    def get_input_lengths(self, src_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if src_padding_mask is not None:
            return torch.sum(torch.logical_not(src_padding_mask), dim=1)
        return (src_ids != self.encoder.tokenizer.pad_token_id).sum(1)


class DecoderMixin:
    decoder: BaseDecoder

    def decode(self,
               tgt: torch.Tensor,
               memory: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None,
               memory_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()


class InferenceMixin:
    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> object:
        raise NotImplementedError()
