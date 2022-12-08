from typing import List, Optional
import einops
import torch
from torch import nn

# copied and modified from https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[List[float]],
        gamma: float,
        reduction: str = "mean",
        ignore_index: int = -100
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(
            weight=torch.as_tensor(alpha, dtype=torch.float) if alpha is not None else None,
            reduction="none",
            ignore_index=ignore_index
        )

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert outputs.ndim == 2 and labels.ndim == 1
        unignored_mask = labels != self.ignore_index
        labels = labels[unignored_mask]
        if len(labels) == 0:
            return torch.tensor(0, device=outputs.device, dtype=torch.float)
        outputs = outputs[unignored_mask]

        log_p = torch.log_softmax(outputs, dim=-1)
        ce = self.nll_loss(log_p, labels)

        log_pt = log_p[torch.arange(len(outputs), device=outputs.device), labels]

        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


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
