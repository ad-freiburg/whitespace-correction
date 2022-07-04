from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from whitespace_correction.model.decoder import BaseDecoder
from whitespace_correction.model.encoder import BaseEncoder


class EncoderMixin:
    def __init__(self) -> None:
        super().__init__()
        # set encoder in child class
        self.encoder: BaseEncoder

    def encode(self,
               src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def get_memory_key_padding_mask(self, src: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class DecoderMixin:
    def __init__(self) -> None:
        super().__init__()
        # set decoder in child class
        self.decoder: BaseDecoder

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
