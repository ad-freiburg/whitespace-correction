from typing import Any, Dict, List

import einops

import torch
from torch import nn

from whitespace_correction.utils import inference
from whitespace_correction.utils.config import HeadConfig


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def inference(
            self,
            encodings: torch.Tensor,
            input_lengths: torch.Tensor,
            **kwargs: Dict[str, Any]
    ) -> List[inference.InferenceResult]:
        raise NotImplementedError()


class ClassificationHead(Head):
    def __init__(self, model_dim: int, num_classes: int, num_layers: int = 1):
        super().__init__()
        assert num_classes > 1, f"number of classes must be greater 1, but got {num_classes}"
        self.model_dim = model_dim
        self.num_classes = num_classes
        self.num_layers = num_layers

        layers = []
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                layers.append(nn.Linear(self.model_dim, self.model_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            else:
                # final linear layer
                layers.append(nn.Linear(self.model_dim, self.num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        :param encodings: tensor of shape [S, B, D]
        :return: tensor of shape [B, C]
        """
        x = encodings[0, :, :]
        x = self.head(x)
        return x

    def inference(
            self,
            encodings: torch.Tensor,
            input_lengths: torch.Tensor,
            **kwargs: Dict[str, Any]
    ) -> List[inference.ClassificationInferenceResult]:
        with torch.no_grad():
            head_outputs = self.forward(encodings)

        if "no_spaces" in kwargs:
            temperatures, thresholds_and_defaults = inference.get_temperatures_thresholds_and_defaults(**kwargs)
        else:
            temperatures = torch.ones(len(head_outputs), dtype=torch.float)
            thresholds_and_defaults = None

        predictions = inference.class_predictions_from_logits(
            logits=head_outputs,
            temperatures=temperatures,
            thresholds_and_defaults=thresholds_and_defaults
        )

        return [
            inference.ClassificationInferenceResult(
                prediction=pred,
                logits=logits
            )
            for pred, logits in zip(predictions, head_outputs.tolist())
        ]


class SequenceClassificationHead(ClassificationHead):
    def __init__(self, model_dim: int, num_classes: int, num_layers: int = 1):
        super().__init__(model_dim=model_dim, num_classes=num_classes, num_layers=num_layers)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        :param encodings: tensor of shape [S, B, D]
        :return: tensor of shape [S, B, C]
        """
        x = self.head(encodings)
        return x

    def inference(
            self,
            encodings: torch.Tensor,
            input_lengths: torch.Tensor,
            **kwargs: Dict[str, Any]
    ) -> List[inference.SequenceClassificationInferenceResult]:
        with torch.no_grad():
            head_outputs = self.forward(encodings)

        # batch first
        head_outputs = einops.rearrange(head_outputs, "s b c -> b s c")
        if "no_spaces" in kwargs:
            temperatures, thresholds_and_defaults = inference.get_temperatures_thresholds_and_defaults(**kwargs)
        else:
            temperatures = torch.ones(len(head_outputs), dtype=torch.float)
            thresholds_and_defaults = None

        predictions = inference.class_predictions_from_logits(
            logits=head_outputs,
            temperatures=temperatures,
            thresholds_and_defaults=thresholds_and_defaults
        )

        return [
            inference.SequenceClassificationInferenceResult(
                predictions=pred[:length],
                logits=logits[:length]
            )
            for pred, logits, length in zip(predictions, head_outputs.tolist(), input_lengths.tolist())
        ]


def get_head_from_config(config: HeadConfig, model_dim: int) -> Head:
    kwargs = config.arguments

    if config.type == "classification":
        head = ClassificationHead(model_dim=model_dim,
                                  num_classes=kwargs["num_classes"],
                                  num_layers=kwargs.get("num_layers", 1))

    elif config.type == "sequence_classification":
        head = SequenceClassificationHead(model_dim=model_dim,
                                          num_classes=kwargs["num_classes"],
                                          num_layers=kwargs.get("num_layers", 1))

    else:
        raise ValueError(f"Unknown head type {config.type}")

    return head
