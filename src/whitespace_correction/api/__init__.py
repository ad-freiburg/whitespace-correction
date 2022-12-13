import collections
import math
import os
import pickle
import pprint
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
from torch import autocast

from tqdm import tqdm
from whitespace_correction import model

from whitespace_correction.model import get_model_from_config
from whitespace_correction.utils import config
from text_correction_utils import inference, api, logging, configuration, io, data

__all__ = ["ModelInfo", "get_available_models", "WhitespaceCorrector"]

ModelInfo = collections.namedtuple("ModelInfo", ["name", "description"])


def get_available_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            name="eo_large",
            description="Best overall model, use this for text that might have OCR or spelling errors."
        ),
        ModelInfo(
            name="eo_medium",
            description="Compromise between eo_large and eo_small, "
                        "faster than eo_large but less accurate, "
                        "slower than eo_small but more accurate."
        ),
        ModelInfo(
            name="eo_small",
            description="Smallest and fastest, but also the least accurate model. "
                        "Use this when you want to repair text with few whitespace errors and "
                        "little to no OCR or spelling errors fast."
        ),
        ModelInfo(
            name="ed_large",
            description="Encoder-decoder model that is similar to eo_large in size."
        ),
        ModelInfo(
            name="ed_medium",
            description="Encoder-decoder model that is similar to eo_medium in size."
        ),
        ModelInfo(
            name="ed_small",
            description="Encoder-decoder model that is similar to eo_small in size."
        )
    ]


StringInputOutput = Union[str, List[str]]


class WhitespaceCorrector:
    @staticmethod
    def from_pretrained(
            model: str = get_available_models()[0].name,
            device: Union[str, int] = "cuda",
            download_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "WhitespaceCorrector":
        assert any(model == m.name for m in get_available_models()), \
            f"model {model} does not match any of the available models:\n{pprint.pformat(get_available_models())}"

        logger = common.get_logger("DOWNLOAD")
        model_dir = download_model(model, download_dir, cache_dir, force_download, logger)

        return WhitespaceCorrector(model_dir, device)

    @staticmethod
    def from_experiment(
            experiment_dir: str,
            device: Union[str, int] = "cuda"
    ) -> "WhitespaceCorrector":
        return WhitespaceCorrector(experiment_dir, device)

    @property
    def model_name(self) -> str:
        return self.cfg.model.name

    def __init__(self,
                 model_dir: str,
                 device: Union[str, int]) -> None:
        self.logger = logging.get_logger("WHITESPACE_CORRECTION")

        if device != "cpu" and not torch.cuda.is_available():
            self.logger.info("could not find a GPU, using CPU as fallback option")
            device = "cpu"

        self.device = torch.device(device)

        self.cfg = configuration.load_config(os.path.join(model_dir, "config.yaml"))
        assert self.cfg["model"]["type"] == "encoder_with_head", \
            "this API currently supports only models of type 'encoder_with_head'"
        self.logger.debug(f"loaded model config:\n{self.cfg['model']}")
        experiment_name = self.cfg["experiment"]["name"]
        self.logger.info(f"running {experiment_name} whitespace corrector on device {api.device_info(self.device)}")

        self.model = model.get_model_from_config()
        best_checkpoint_path = os.path.join(model_dir, "checkpoints", "checkpoint_best.pt")
        best_checkpoint = io.load_checkpoint(best_checkpoint_path)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.input_tokenizer = config.get_tokenizer_from_config(self.cfg["input_tokenizer"])

        # use the training max sequence length here, even though some models work with arbitrary long sequences
        # (e.g. LSTM), for better accuracy
        self.max_length = self.cfg["model"]["embedding"]["max_length"] \
            - self.input_tokenizer.num_prefix_tokens() - self.input_tokenizer.num_suffix_tokens()
        self.window_size = math.ceil(0.75 * self.max_length)
        self.context_size = (self.max_length - self.window_size) // 2

        self._mixed_precision_dtype = torch.float32

    @torch.inference_mode()
    def _repair_text_raw(
            self,
            inputs: List[str],
            inputs_are_files: bool,
            languages: Optional[List[str]],
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False,
            num_threads: int = 0
    ) -> List[inference.WhitespaceCorrectionInferenceResult]:
        if inputs_are_files:
            loader = data.DataLoader.from_files(
                inputs,
                languages=languages,
                num_threads=num_threads
            )
        else:
            loader = data.DataLoader.from_sequences(
                inputs,
                languages=languages,
                num_threads=num_threads
            )
        loader = iter(loader)
        pbar = tqdm(
            loader,
            total=sum_lengths,
            ascii=True,
            leave=False,
            disable=not show_progress,
            unit="char"
        )

        for i, batch_idx in enumerate(pbar):
            batch = batches[batch_idx: batch_idx + batch_size]
            batch_sequences = [
                inputs[input_idx][from_:to_]
                for input_idx, from_, to_, _ in batch
            ]
            batch_length = sum(len(s) for s in batch_sequences)

            pbar.set_description(
                f"[Batch {i + 1}] Repairing whitespaces of {len(batch):,} sequences"
            )

            # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
            if self.mixed_precision_enabled:
                with autocast(
                        device_type=self.device.type,
                        dtype=self._mixed_precision_dtype,
                        enabled=self.mixed_precision_enabled
                ):
                    batch_inference_results = self.model.inference(
                        batch_sequences,
                        **kwargs
                    )
            else:
                batch_inference_results = self.model.inference(
                    batch_sequences,
                    **kwargs
                )
            for (input_idx, _, _, position), inference_result in zip(
                    batch,
                    batch_inference_results
            ):
                all_inference_results[input_idx][position] = inference_result  # type: ignore

            pbar.update(batch_length)

        pbar.close()
        return [
            self._merge_inference_results(all_inference_results[i], inputs[i])  # type: ignore
            for i in range(len(inputs))
        ]

    def correct_text(
            self,
            inputs: StringInputOutput,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> StringInputOutput:
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), "input needs to be a string or a non empty list of strings"

        if input_is_string:
            inputs = [inputs]  # type: ignore

        # clean inputs from leading, trailing or multiple whitespaces
        inputs = [whitespace_correction.clean_sequence(ipt) for ipt in inputs]

        inference_results = self._repair_text_raw(inputs, batch_size, sort_by_length, show_progress)

        outputs = [self._inference_result_to_str(ir, ipt) for ipt, ir in zip(inputs, inference_results)]
        return outputs[0] if input_is_string else outputs

    def correct_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[List[str]]:
        inputs = []
        with open(input_file_path, "r", encoding="utf8") as in_file:
            for line in in_file:
                inputs.append(line.strip())

        outputs = self.correct_text(inputs, batch_size, sort_by_length, show_progress)

        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(output + "\n")
            return None
        else:
            return outputs  # type: ignore

    def to(self, device: Union[str, int]) -> "WhitespaceCorrector":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def set_precision(self, precision: str) -> None:
        assert precision in {"fp32", "fp16", "bfp16"}

        if precision == "fp32":
            mixed_precision_dtype = torch.float32
        elif precision == "fp16":
            mixed_precision_dtype = torch.float16
        else:
            mixed_precision_dtype = torch.bfloat16

        if self.device.type == "cpu" and precision == "fp16":
            self.logger.info("Setting precision to bfp16 instead of fp16, because fp16 is not supported on CPU yet")
            mixed_precision_dtype = torch.bfloat16

        self._mixed_precision_dtype = mixed_precision_dtype

    @property
    def mixed_precision_enabled(self) -> bool:
        return self._mixed_precision_dtype != torch.float32
