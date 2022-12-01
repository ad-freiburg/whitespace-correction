import einops
import torch
from torch import nn


class SeqLoss(nn.Module):
    """
    Wrapper class for sequence losses. Rearranges outputs and labels to use with standard Pytorch losses.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # outputs are expected to be of shape [B, S, C], reshape to [B * S, C]
        outputs = einops.rearrange(outputs, "b s c -> (b s) c")
        # labels are expected to be of shape [B, S], reshape to [B * S]
        labels = einops.rearrange(labels, "b s -> (b s)")
        return self.loss(outputs, labels)
