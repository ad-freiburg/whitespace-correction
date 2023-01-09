import collections
import math
from typing import Any, Dict, List, Optional, Union, Iterator

import torch
from torch import nn
from torch import autocast

from whitespace_correction.model import model_from_config

from text_correction_utils import data, whitespace, tokenization
from text_correction_utils.api import corrector, device_info, ModelInfo


class WhitespaceCorrector(corrector.TextCorrector):
    task = "whitespace correction"

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="eo_large",
                description="Best overall model, use this for text that might have OCR or spelling errors.",
                tags=[]
            ),
            ModelInfo(
                name="eo_medium",
                description="Compromise between eo_large and eo_small, "
                "faster than eo_large but less accurate, "
                "slower than eo_small but more accurate.",
                tags=[]
            ),
            ModelInfo(
                name="eo_small",
                description="Smallest and fastest, but also the least accurate model. "
                "Use this when you want to repair text with few whitespace errors and "
                "little to no OCR or spelling errors fast.",
                tags=[]
            ),
            ModelInfo(
                name="ed_large",
                description="Encoder-decoder model that is similar to eo_large in size.",
                tags=[]
            ),
            ModelInfo(
                name="ed_medium",
                description="Encoder-decoder model that is similar to eo_medium in size.",
                tags=[]
            ),
            ModelInfo(
                name="ed_small",
                description="Encoder-decoder model that is similar to eo_small in size.",
                tags=[]
            )
        ]

    @classmethod
    def _model_url(cls, model: str) -> str:
        _BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
            "EMNLP_whitespace_correction_transformer_BHW_2022.materials"
        NAME_TO_URL = {
            "eo_large": f"{_BASE_URL}/eo_large.zip",
            "eo_small": f"{_BASE_URL}/eo_small.zip",
            "eo_medium": f"{_BASE_URL}/eo_medium.zip",
            "ed_large": f"{_BASE_URL}/ed_large.zip",
            "ed_medium": f"{_BASE_URL}/ed_medium.zip",
            "ed_small": f"{_BASE_URL}/ed_small.zip",
        }
        return NAME_TO_URL[model]

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        input_tokenizer = tokenization.tokenizer_from_config(cfg["input_tokenizer"])
        return model_from_config(
            cfg["model"],
            input_tokenizer,
            None
        )

    @property
    def max_length(self) -> int:
        return self.cfg["model"]["embedding"]["max_length"]

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    def __init__(
        self,
            model_dir: str,
            device: Union[str, int]
    ) -> None:
        super().__init__(model_dir, device)
        precision = self.cfg["train"].get("mixed_precision_dtype", "fp32")
        self.set_precision(precision)
        assert (
            self.cfg["model"]["type"] == "encoder_with_head"
            and self.cfg["model"]["head"]["type"] == "sequence_classification"
        ), "this API currently supports only models of type 'encoder_with_head' with a sequence classification head"
        self.logger.debug(f"loaded model config:\n{self.cfg['model']}")
        self.logger.info(f"running {self.name} whitespace corrector on device {device_info(self.device)}")
        self.input_tokenizer = config.get_tokenizer_from_config(self.cfg["input_tokenizer"])
        self._pfx = self.input_tokenizer.num_prefix_tokens()
        self._sfx = self.input_tokenizer.num_suffix_tokens()

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        input_tokenizer_cfg = config.get_tokenizer_config(self.cfg["input_tokenizer"])
        input_tokenizer = config.get_tokenizer_from_config(self.cfg["input_tokenizer"])
        pfx = input_tokenizer.num_prefix_tokens()
        sfx = input_tokenizer.num_suffix_tokens()

        # use the training max sequence length here, even though some models work with arbitrary long sequences
        # (e.g. LSTM), for better accuracy
        max_length = self.max_length - pfx - sfx
        window_size = math.ceil(0.75 * max_length)
        context_size = (max_length - window_size) // 2
        if self.cfg["input_tokenizer"]["tokenize"]["type"] == "byte":
            window_cfg = {"type": "byte", "max_bytes": max_length, "context_bytes": context_size}
        elif self.cfg["input_tokenizer"]["tokenize"]["type"] == "character":
            window_cfg = {"type": "character", "max_chars": max_length, "context_chars": context_size}
        else:
            raise ValueError("the input tokenizer must be of type 'char' or 'byte' for whitespace correction")

        # this config should be also used during training whitespace correciton models
        return {
            "tokenizer_config": input_tokenizer_cfg,
            "window_config": window_cfg,
        }

    @staticmethod
    def _prepare_info(
        info: Dict[str, Any]
    ) -> Dict[str, Any]:
        info_type = info.pop("type")
        if info_type in {"empty", "token_groups"}:
            return info
        else:
            raise ValueError(f"unknown info type {info_type}")

    def _prepare_batch(self, batch: data.InferenceItemBatch) -> Dict[str, Any]:
        if self.cfg["model"]["type"] == "encoder_with_head":
            pad_token_id = self.input_tokenizer.pad_token_id()
            token_ids = []
            token_lengths = [len(item.tokenization.token_ids) for item in batch]
            max_tokens = max(token_lengths)
            kwargs = collections.defaultdict(list)
            for i, item in enumerate(batch):
                token_ids.append(item.tokenization.token_ids + [pad_token_id] * (max_tokens - token_lengths[i]))
                for k, v in self._prepare_info(item.tokenization.info).items():
                    kwargs[k].append(v)
            inputs = {
                "token_ids": torch.as_tensor(token_ids, dtype=torch.long, device=self.device),
                "lengths": token_lengths,
                **kwargs
            }
            return inputs
        else:
            raise ValueError(f"unsupported model type {self.cfg['model']['type']}")

    def _process_results(self, items: List[data.InferenceItem], outputs: List[Any]) -> str:
        assert len(items) > 0 and len(items) == len(outputs)
        merged_predictions = []
        for item, output in zip(items, outputs):
            context_start, window_start, window_end, _ = item.window
            window_start -= context_start
            window_end -= context_start
            prediction = torch.argmax(output[self._pfx + window_start:self._pfx + window_end], dim=-1)
            merged_predictions.extend(prediction.tolist())
        return whitespace.repair(
            items[0].data.original,
            merged_predictions
        )

    def correct_text(
            self,
            inputs: Union[str, List[str]],
            languages: Optional[List[str]] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None
    ) -> Union[str, List[str], Iterator[str]]:
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), "input needs to be a string or a list of strings"

        if input_is_string:
            inputs = [inputs]  # type: ignore
            if languages is not None:
                assert isinstance(languages, str), "language must be a string if specified and input is a string"
                languages = [languages]
        loader = self._get_loader(
            inputs,
            "sequences",
            languages,
            batch_size,
            batch_max_tokens,
            sort,
            num_threads
        )
        if sort:
            outputs = self._correct_sorted(loader)
            return outputs[0] if input_is_string else outputs
        else:
            return self._correct_unsorted(loader)

    def correct_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            language: Optional[str] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None
    ) -> Optional[Union[Iterator[str], List[str]]]:
        loader = self._get_loader(
            [input_file_path],
            "files",
            [language] if language is not None else None,
            batch_size,
            batch_max_tokens,
            sort,
            num_threads
        )
        if sort:
            outputs = self._correct_sorted(loader)
        else:
            outputs = self._correct_unsorted(loader)
        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as of:
                for output in outputs:
                    of.write(output + "\n")
        else:
            return outputs

    def set_precision(self, precision: str) -> None:
        training_precision = self.cfg["train"].get("mixed_precision_dtype", "fp32")
        if precision != "fp32" and precision != training_precision:
            self.logger.warning(
                f"this model was trained with {training_precision} precision, "
                "inference with {precision} might give unexpected results"
            )
        return super().set_precision(precision)
