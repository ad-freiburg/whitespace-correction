from io import TextIOWrapper
import math
import os
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator

import torch
from torch import nn
from torch import autocast

from whitespace_correction.model import model_from_config, EncoderDecoderWithHead

from text_correction_utils import data, whitespace, tokenization
from text_correction_utils.api.corrector import ModelInfo
from text_correction_utils.api import corrector
from text_correction_utils.api.utils import device_info, to
from text_correction_utils.inference import IdxSelectFn, eos_stop_fn, search

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "EMNLP_whitespace_correction_transformer_BHW_2022.materials"
_NAME_TO_ZIP = {
    "eo_large_v1": "eo_large_v1.zip",
    "eo_large_v2": "eo_large_v2.zip",
    "eo_medium_v1": "eo_medium_v1.zip",
    "eo_medium_v2": "eo_medium_v2.zip",
    # "eo_small": "eo_small.zip",
    "ed_large_v1": "ed_large_v1.zip",
    "ed_medium_v1": "ed_medium_v1.zip",
    # "ed_small": "ed_small.zip",
}


class WhitespaceCorrector(corrector.TextCorrector):
    task = "whitespace correction"

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="eo_large_v2",
                description="Combines fast inference and good performance",
                tags=["default", "lang::en", "arch::encoder-only", "input::byte"]
            ),
            ModelInfo(
                name="eo_large_v1",
                description="Combines fast inference and good performance",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="eo_medium_v2",
                description="Encoder-only model; smaller and faster than eo_large_v2, but less accurate",
                tags=["lang::en", "arch::encoder-only", "input::byte"]
            ),
            ModelInfo(
                name="eo_medium_v1",
                description="Encoder-only model; smaller and faster than eo_large_v1, but less accurate",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="ed_large_v1",
                description="Similar to eo_large_v1 in size",
                tags=["lang::en", "arch::encoder-decoder", "input::char", "output::char"]
            ),
            ModelInfo(
                name="ed_medium_v1",
                description="Smaller and faster than ed_large_v1, but less accurate",
                tags=["lang::en", "arch::encoder-decoder", "input::char", "output::char"]
            ),
        ]

    @classmethod
    def _model_url(cls, model: str) -> str:
        return f"{_BASE_URL}/{_NAME_TO_ZIP[model]}"

    @property
    def name(self) -> str:
        return self.cfg["experiment"]["name"]

    @classmethod
    def _model_from_config(cls, cfg: Dict[str, Any]) -> nn.Module:
        input_tokenizer = tokenization.Tokenizer.from_config(cfg["input_tokenizer"])
        if "output_tokenizer" in cfg:
            output_tokenizer = tokenization.Tokenizer.from_config(cfg["output_tokenizer"])
        else:
            output_tokenizer = None
        return model_from_config(
            cfg["model"],
            input_tokenizer,
            output_tokenizer
        )

    @property
    def max_length(self) -> int:
        if self.cfg["model"]["type"] == "pretrained_encoder_with_head":
            return 512
        elif self.cfg["model"]["type"] == "encoder_with_head":
            return self.cfg["model"]["embedding"].get("max_length", 512)
        elif self.cfg["model"]["type"] == "encoder_decoder_with_head":
            return self.cfg["model"]["encoder_embedding"].get("max_length", 512)
        else:
            raise ValueError(f"unknown model type: {self.cfg['model']['type']}")

    @property
    def context_length(self) -> int:
        raise NotImplementedError

    @property
    def supported_languages(self) -> Optional[List[str]]:
        lang_cfg = self.cfg["input_tokenizer"].get("language")
        if lang_cfg is None:
            return None
        else:
            return lang_cfg["languages"]

    def __init__(
        self,
            model_dir: str,
            device: Union[str, int]
    ) -> None:
        super().__init__(model_dir, device)
        precision = self.cfg["train"].get("mixed_precision_dtype", "fp32")
        self.set_precision(precision)
        self.logger.debug(f"loaded model config:\n{self.cfg['model']}")
        self.logger.info(f"running {self.name} whitespace corrector on device {device_info(self.device)}")
        self.input_tokenizer = tokenization.Tokenizer.from_config(self.cfg["input_tokenizer"])
        if "output_tokenizer" in self.cfg:
            self.output_tokenizer = tokenization.Tokenizer.from_config(self.cfg["output_tokenizer"])
        else:
            self.output_tokenizer = None

        self._encoder_only = self.cfg["model"]["type"].endswith("encoder_with_head")
        self._pfx = self.input_tokenizer.num_prefix_tokens()
        self._sfx = self.input_tokenizer.num_suffix_tokens()

    def _build_inference_loader_config(self) -> Dict[str, Any]:
        input_tokenizer = tokenization.Tokenizer.from_config(self.cfg["input_tokenizer"])
        pfx = input_tokenizer.num_prefix_tokens()
        sfx = input_tokenizer.num_suffix_tokens()

        # use the training max sequence length here, even though some models work with arbitrary long sequences
        # (e.g. LSTM), for better accuracy
        max_length = self.max_length - pfx - sfx
        window_size = math.ceil(0.75 * max_length)
        context_size = (max_length - window_size) // 2
        if self.cfg["input_tokenizer"]["tokenize"]["type"] in {"byte", "byt5"}:
            window_cfg = {"type": "byte", "max_bytes": max_length, "context_bytes": context_size}
        elif self.cfg["input_tokenizer"]["tokenize"]["type"] == "character":
            window_cfg = {"type": "character", "max_chars": max_length, "context_chars": context_size}
        else:
            raise ValueError("the input tokenizer must be of type 'char' or 'byte' for whitespace correction")

        return {
            "tokenizer_config": self.cfg["input_tokenizer"],
            "window_config": window_cfg,
        }

    def _prepare_batch(self, batch: data.InferenceBatch) -> Dict[str, Any]:
        token_ids_np, pad_mask_np, lengths, info = batch.tensors
        inputs = {
            "token_ids": torch.from_numpy(token_ids_np).to(non_blocking=True, device=self.device),
            "padding_mask": torch.from_numpy(pad_mask_np).to(non_blocking=True, device=self.device),
            "lengths": lengths,
            **to(info, self.device)
        }
        return inputs

    def _inference(self, inputs: Dict[str, Any]) -> Any:
        if self._encoder_only:
            outputs, _ = self.model(**inputs)
            return outputs

        assert isinstance(self.model, EncoderDecoderWithHead)
        encoded, kwargs = self.model.encode(**inputs)

        # decode fn gets in token ids and additional kwargs,
        # and return logits over next tokens
        def _decode_fn(
            token_ids: torch.Tensor,
            **kwargs: Any
        ) -> torch.Tensor:
            decoded = self.model.decode(
                token_ids,
                **kwargs
            )
            return decoded

        def _kwargs_sub_select_fn(kwargs: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
            return {
                "memories": {"encoder": kwargs["memory"][mask]},
                "memory_padding_masks": {"encoder": kwargs["padding_mask"][mask]}
            }

        max_output_length = self.cfg["model"]["decoder_embedding"].get("max_length", 2 * self.max_length)

        # use a custom select function that only allows select
        # the whitespace token or the copy the corrpesonding token from
        # the input
        assert self.output_tokenizer is not None
        eos_token_id = self.output_tokenizer.eos_token_id()
        ws_token_id = self.output_tokenizer.tokenize(" ").token_ids[self._pfx]
        lengths = inputs["lengths"]
        non_ws_token_ids = [
            [t for t in token_ids[self._pfx: length - self._sfx] if t != ws_token_id]
            for token_ids, length in zip(inputs["token_ids"].tolist(), lengths)
        ]
        token_id_indices = [0] * len(non_ws_token_ids)
        last_was_ws = [False] * len(non_ws_token_ids)

        # this select funciton makes sure that we
        # either copy the correct input char or add a whitespace,
        # also makes sure that no doubled whitespaces are returned
        def _custom_select_fn() -> IdxSelectFn:
            def _select(scores: torch.Tensor, idx: int) -> Tuple[int, float]:
                token_id_idx = token_id_indices[idx]
                if token_id_idx >= len(non_ws_token_ids[idx]):
                    # we are at the end of the input, select eos
                    return eos_token_id, 0

                input_token_id = non_ws_token_ids[idx][token_id_idx]
                ws_score = scores[ws_token_id]
                input_token_score = scores[input_token_id]
                if ws_score > input_token_score and not last_was_ws[idx]:
                    last_was_ws[idx] = True
                    return ws_token_id, float(ws_score)
                else:
                    token_id_indices[idx] += 1
                    last_was_ws[idx] = False
                    return input_token_id, float(input_token_score)

            return _select

        output = search(
            decode_fn=_decode_fn,
            initial_token_ids=[[self.output_tokenizer.bos_token_id()]] * encoded.shape[0],
            pad_token_id=self.output_tokenizer.pad_token_id(),
            max_length=max_output_length,
            select_fn=_custom_select_fn(),
            stop_fn=eos_stop_fn(self.output_tokenizer.eos_token_id()),
            device=self.device,
            kwargs_sub_select_fn=_kwargs_sub_select_fn,
            memory=encoded,
            **kwargs
        )
        return output

    def _process_results(
        self,
        items: List[data.InferenceItem],
        outputs: List[Any],
    ) -> data.InferenceData:
        assert len(items) > 0 and len(items) == len(outputs)
        if self._encoder_only:
            merged_predictions = []
            for item, output in zip(items, outputs):
                context_start, window_start, window_end, _ = item.window
                window_start -= context_start
                window_end -= context_start
                prediction = torch.argmax(output[self._pfx + window_start:self._pfx + window_end], dim=-1)
                merged_predictions.extend(prediction.tolist())
            repaired = whitespace.repair(items[0].data.text, merged_predictions)
            return data.InferenceData(repaired, language=items[0].data.language)

        # only thing left to do here is swap back the unknown tokens
        # with the original ones
        assert self.output_tokenizer is not None
        unk_token_id = self.output_tokenizer.unk_token_id()
        out_pfx = self.output_tokenizer.num_prefix_tokens()
        out_sfx = self.output_tokenizer.num_suffix_tokens()
        merged = ""
        for item, output in zip(items, outputs):
            context_start, window_start, window_end, _ = item.window
            window_start -= context_start
            window_end -= context_start
            input_chars = item.context_chars()
            input_token_ids = item.tokenization.token_ids[self._pfx:-self._sfx]
            assert len(input_chars) == len(input_token_ids)
            input_unk_indices = [
                i
                for i, tok_id in enumerate(input_token_ids)
                if tok_id == unk_token_id
            ]
            input_unk_chars = [input_chars[i] for i in input_unk_indices]
            output_token_ids = output[out_pfx:-out_sfx]
            output_unk_indices = [
                i
                for i, tok_id in enumerate(output_token_ids)
                if tok_id == unk_token_id
            ]
            assert len(input_unk_indices) == len(output_unk_indices)
            output_str = ""
            start_idx = 0
            for output_unk_idx, input_unk_char in zip(output_unk_indices, input_unk_chars):
                output_str += self.output_tokenizer.de_tokenize(output_token_ids[start_idx:output_unk_idx])
                start_idx = output_unk_idx + 1
                output_str += input_unk_char
            output_str += self.output_tokenizer.de_tokenize(output_token_ids[start_idx:])
            window_str = item.window_str()
            output_str = whitespace.find_substring_ignoring_whitespace(output_str, window_str)
            assert output_str is not None
            merged += output_str.lstrip()

        return data.InferenceData(merged.rstrip(), language=items[0].data.language)

    def correct_text(
            self,
            inputs: Union[str, List[str]],
            languages: Optional[List[str]] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            show_progress: bool = False
    ) -> Union[str, List[str]]:
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), "input needs to be a string or a list of strings"

        if input_is_string:
            inputs = [inputs]

        if languages is not None:
            if input_is_string:
                assert isinstance(languages, str), "language must be a string if specified and input is a string"
                langs = [languages]
            else:
                assert (
                    isinstance(languages, list)
                    and all(isinstance(lang, str) for lang in languages)
                    and len(languages) == len(inputs)
                ), "expected same number of languages as inputs"
                langs = languages
        else:
            langs = [None] * len(inputs)

        loader = self._get_loader(
            (data.InferenceData(s, language=l) for s, l in zip(inputs, langs)),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = f"Correcting whitespaces in {len(inputs)} sequences"
        progress_total = len(inputs)
        progress_unit = "seq"

        if sort:
            outputs = self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        return next(iter(outputs)).text if input_is_string else [output.text for output in outputs]

    def correct_iter(
        self,
        iter: Iterator[Tuple[str, Optional[str]]],
        batch_size: int = 16,
        batch_max_tokens: Optional[int] = None,
        sort: bool = True,
        num_threads: Optional[int] = None,
        return_raw: bool = False,
        show_progress: bool = False
    ) -> Union[Iterator[str], Iterator[data.InferenceData]]:
        loader = self._get_loader(
            (data.InferenceData(s, language=l) for s, l in iter),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
        )

        progress_desc = "Correcting whitespaces in iterator"
        progress_total = sys.maxsize
        progress_unit = "byte"

        if sort:
            output = self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            output = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if return_raw:
            yield from output
        else:
            yield from (data.text for data in output)

    def correct_file(
            self,
            input_file: str,
            input_file_format: str = "text",
            output_file: Optional[Union[TextIOWrapper, str]] = None,
            output_file_format: str = "text",
            language: Optional[str] = None,
            batch_size: int = 16,
            batch_max_tokens: Optional[int] = None,
            sort: bool = True,
            num_threads: Optional[int] = None,
            show_progress: bool = False
    ) -> Optional[Iterator[str]]:
        assert input_file_format in self.supported_input_formats(), f"unsupported input file format {input_file_format}, \
        must be one of {self.supported_input_formats()}"
        assert output_file_format in self.supported_output_formats(), f"unsupported output file format {output_file_format}, \
        must be one of 'text' or 'text_language'"
        loader = self._get_loader(
            ([input_file], [language] if language is not None else None),
            batch_size,
            batch_max_tokens,
            sort,
            num_threads,
            file_format=input_file_format,
        )

        file_name = input_file if len(input_file) < 32 else f"...{input_file[-29:]}"
        progress_desc = f"Correcting whitespaces in {file_name}"
        progress_total = os.path.getsize(input_file)
        progress_unit = "byte"

        if sort:
            outputs = iter(self._correct_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            ))
        else:
            outputs = self._correct_unsorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )

        if output_file is not None:
            output_file_is_str = isinstance(output_file, str)
            if output_file_is_str:
                output_dir = os.path.dirname(output_file)
                if output_dir != "":
                    os.makedirs(output_dir, exist_ok=True)
                output_file = open(output_file, "w", encoding="utf8")

            for output in outputs:
                output_file.write(f"{output.to_str(output_file_format)}\n")

            if output_file_is_str:
                output_file.close()

        else:
            return (output.text for output in outputs)

    def set_precision(self, precision: str) -> None:
        training_precision = self.cfg["train"].get("mixed_precision_dtype", "fp32")
        if precision != "fp32" and precision != training_precision:
            self.logger.warning(
                f"this model was trained with {training_precision} precision, "
                "inference with {precision} might give unexpected results"
            )
        return super().set_precision(precision)
