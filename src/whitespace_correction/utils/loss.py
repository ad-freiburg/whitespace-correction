from typing import Any

import torch
from torch import nn

from whitespace_correction.utils.config import LossConfig


def get_loss_from_config(config: LossConfig,
                         vocab_size: int,
                         ignore_index: int) -> nn.Module:
    loss_kwargs = {"ignore_index": ignore_index,
                   "vocab_size": vocab_size}
    # add arguments from config
    loss_kwargs.update(config.arguments)
    criterion = get_loss(name=config.type,
                         **loss_kwargs)
    return criterion


def get_loss(name: str,
             **kwargs: Any) -> nn.Module:
    if name == "seq2seq_cross_entropy":
        weight = kwargs.get("weight", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None

        loss = nn.CrossEntropyLoss(ignore_index=kwargs["ignore_index"], weight=weight)
        return Seq2SeqLoss(loss=loss)

    elif name == "seq2seq_label_smoothed_cross_entropy":
        loss = LabelSmoothingLogitsLoss(label_smoothing=kwargs["label_smoothing"],
                                        tgt_vocab_size=kwargs["vocab_size"],
                                        ignore_index=kwargs["ignore_index"])
        return Seq2SeqLoss(loss=loss)

    elif name == "cross_entropy":
        weight = kwargs.get("weight", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.CrossEntropyLoss(ignore_index=kwargs.get("ignore_index", -100), weight=weight)
        return loss

    elif name == "binary_cross_entropy":
        weight = kwargs.get("weight", None)
        weight = torch.tensor(weight, dtype=torch.float) if weight is not None else None
        loss = nn.BCELoss(weight=weight)
        return loss

    else:
        raise ValueError(f"Unknown loss {name}")


class Seq2SeqLoss(nn.Module):
    """
    Wrapper class for Seq2Seq losses. Reshapes outputs and labels to use with standard Pytorch losses.
    """

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # outputs are expected to be of shape [S, B, C], reshape to [B * S, C]
        outputs = outputs.transpose(0, 1).reshape(-1, outputs.shape[-1])
        # labels are expected to be of shape [B, S], reshape to [B * S]
        labels = labels.reshape(-1)
        return self.loss(outputs, labels)


class LabelSmoothingLogitsLoss(nn.Module):
    def __init__(self,
                 label_smoothing: float,
                 tgt_vocab_size: int,
                 ignore_index: int):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        model_prob = self.one_hot.repeat(tgt.shape[0], 1)
        model_prob.scatter_(1, tgt.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((tgt == self.ignore_index).unsqueeze(1), 0)
        pred_prob = torch.log_softmax(logits, dim=1)
        return torch.kl_div(pred_prob, model_prob)
