from typing import Any, Dict, List, Tuple, Union

import torch

from whitespace_correction.model.decoder import BaseDecoder
from whitespace_correction.model.encoder import BaseEncoder
from whitespace_correction.utils import mask as mask_utils


class EncoderMixin:
    # set encoder in child class
    encoder: BaseEncoder

    def encode(self,
               src: torch.Tensor,
               **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def get_encoder_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        return mask_utils.get_padding_mask_from_token_ids(src, self.encoder.padding_token_id)


class DecoderMixin:
    decoder: BaseDecoder

    def decode(self,
               tgt: torch.Tensor,
               **kwargs: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()


class InferenceMixin:
    def inference(self,
                  sequences: Union[str, List[str]],
                  **kwargs: Any) -> object:
        raise NotImplementedError()
