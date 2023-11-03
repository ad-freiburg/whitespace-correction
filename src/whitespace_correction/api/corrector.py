from io import TextIOWrapper
import math
import os
import sys
from typing import Any, Dict, List, Tuple, Optional, Union, Iterator

import torch
from torch import nn

from whitespace_correction.model import model_from_config, EncoderDecoderWithHead

from text_utils import data, whitespace, tokenization
from text_utils.api.processor import ModelInfo, TextProcessor
from text_utils.api.utils import device_info, to, Device
from text_utils.inference import IdxSelectFn, eos_stop_fn, search

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
    "ACL_whitespace_correction_transformer_BHW_2023.materials"
_NAME_TO_ZIP = {
    "eo_large_char_v1": "eo_large_v1.zip",
    "eo_large_char": "eo_large_char_v2.zip",
    "eo_large_byte": "eo_large_byte_v2.zip",
    "eo_larger_byte": "eo_huge_byte_v2.zip",
    "eo_medium_char_v1": "eo_medium_v1.zip",
    "eo_medium_char": "eo_medium_char_v2.zip",
    "eo_medium_byte": "eo_medium_byte_v2.zip",
    "ed_large_char": "ed_large_v1.zip",
    "ed_medium_char": "ed_medium_v1.zip",
}


class WhitespaceCorrector(TextProcessor):
    task = "whitespace correction"

    @classmethod
    def available_models(cls) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="eo_large_byte",
                description="Byte-level model combining fast inference and good quality",
                tags=["default", "lang::en", "arch::encoder-only", "input::byte"]
            ),
            ModelInfo(
                name="eo_large_char",
                description="Character-level model combining fast inference and good quality",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="eo_large_char_v1",
                description="Character-level model combining fast inference and good quality, "
                "trained with a different loss than eo_large_char",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="eo_larger_byte",
                description="Larger and slower than eo_large_byte, but also more accuracte",
                tags=["lang::en", "arch::encoder-only", "input::byte"]
            ),
            ModelInfo(
                name="eo_medium_byte",
                description="Smaller and faster than eo_large_byte, but less accurate",
                tags=["lang::en", "arch::encoder-only", "input::byte"]
            ),
            ModelInfo(
                name="eo_medium_char",
                description="Smaller and faster than eo_large_char, but less accurate",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="eo_medium_char_v1",
                description="Smaller and faster than eo_large_char_v1, but less accurate",
                tags=["lang::en", "arch::encoder-only", "input::char"]
            ),
            ModelInfo(
                name="ed_large_char",
                description="Similar to eo_large_byte in size and quality, but slower due to "
                "its autoregressive decoder",
                tags=["lang::en", "arch::encoder-decoder", "input::char", "output::char"]
            ),
            ModelInfo(
                name="ed_medium_char",
                description="Smaller and faster than ed_large, but less accurate",
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
    def _model_from_config(
        cls,
        cfg: Dict[str, Any],
        device: Device
    ) -> nn.Module:
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
        return self.cfg["train"]["data"].get("max_length", 512)

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
        model: nn.Module,
        cfg: Dict[str, Any],
        device: Device = "cuda"
    ) -> None:
        super().__init__(model, cfg, device)
        assert len(self.devices) == 1, \
            "whitespace correction is only supported on single devices for now"
        self.device = self.devices[0]
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
        token_ids_np, pad_mask_np, lengths, info = batch.tensors()
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
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            decoded = self.model.decode(
                token_ids,
                **kwargs
            )
            return decoded, {}

        def _kwargs_select_fn(kwargs: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
            return {
                "memories": {"encoder": kwargs["memory"][mask]},
                "memory_padding_masks": {"encoder": kwargs["memory_padding_mask"][mask]}
            }

        max_output_length = self.cfg["model"]["decoder_embedding"].get("max_length", 2 * self.max_length)

        # use a custom select function that only allows select
        # the whitespace token or the copy the corrpesonding token from
        # the input
        assert self.output_tokenizer is not None
        eos_token = "<eos>"
        eos_token_id = self.output_tokenizer.special_token_to_id(eos_token)
        bos_token = "<bos>"
        bos_token_id = self.output_tokenizer.special_token_to_id(bos_token)
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
            def _select(scores: torch.Tensor, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
                token_ids = []
                log_probs = []
                for dist, idx in zip(scores, indices):
                    token_id_idx = token_id_indices[idx]
                    if token_id_idx >= len(non_ws_token_ids[idx]):
                        # we are at the end of the input, select eos
                        token_ids.append(eos_token_id)
                        log_probs.append(0.0)
                        continue

                    input_token_id = non_ws_token_ids[idx][token_id_idx]
                    ws_score = dist[ws_token_id]
                    input_token_score = dist[input_token_id]
                    if ws_score > input_token_score and not last_was_ws[idx]:
                        last_was_ws[idx] = True
                        token_ids.append(ws_token_id)
                        log_probs.append(float(ws_score))
                    else:
                        token_id_indices[idx] += 1
                        last_was_ws[idx] = False
                        token_ids.append(input_token_id)
                        log_probs.append(float(input_token_score))
                return torch.tensor(token_ids, device=scores.device), torch.tensor(log_probs, device=scores.device)

            return _select

        output = search(
            decode_fn=_decode_fn,
            initial_token_ids=[[bos_token_id]] * encoded.shape[0],
            pad_token_id=self.output_tokenizer.pad_token_id(),
            max_length=max_output_length,
            select_fn=_custom_select_fn(),
            stop_fn=eos_stop_fn(eos_token_id),
            device=self.device,
            kwargs_select_fn=_kwargs_select_fn,
            memory=encoded,
            memory_padding_mask=kwargs["padding_mask"],
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
            return data.InferenceData(
                repaired.strip(),
                language=items[0].data.language
            )

        # only thing left to do here is swap back the unknown tokens
        # with the original ones
        assert self.output_tokenizer is not None
        unk_token = "<unk>"
        unk_token_id = self.output_tokenizer.special_token_to_id(unk_token)
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
            # we only need to exclude the suffix here because by default the search
            # function returns only the newly decoded tokens, which means the prefix is
            # already excluded
            output_token_ids = output[:-out_sfx]
            output_unk_indices = [
                i
                for i, tok_id in enumerate(output_token_ids)
                if tok_id == unk_token_id
            ]
            assert len(input_unk_indices) == len(output_unk_indices)
            output_str = ""
            start_idx = 0
            for output_unk_idx, input_unk_char in zip(output_unk_indices, input_unk_chars):
                output_str += self.output_tokenizer.de_tokenize(
                    output_token_ids[start_idx:output_unk_idx]
                )
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
            outputs = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            outputs = self._process_unsorted(
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
            output = self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            )
        else:
            output = self._process_unsorted(
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
            outputs = iter(self._process_sorted(
                loader,
                progress_desc,
                progress_total,
                progress_unit,
                show_progress
            ))
        else:
            outputs = self._process_unsorted(
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

    def set_precision(self, precision: str) -> TextProcessor:
        if self.pretrained and self.devices[0].type == "cpu" and precision != "fp32":
            self.logger.info(
                f"setting precision to fp32 instead of {precision}, "
                f"because the pretrained {self.task} models do not "
                f"support {precision} on CPU"
            )
            precision = "fp32"
        return super().set_precision(precision)
