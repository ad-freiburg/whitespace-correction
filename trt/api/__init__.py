import collections
import enum
import os
import pickle
import pprint
from typing import Union, List, Optional, Tuple

import torch
from tqdm import tqdm

from trt.api.utils import download_model
from trt.model import transformer

from trt.utils import common, config, io, inference, nlp, tokenization_repair

__all__ = ["ModelInfo", "get_available_models", "TokenizationRepairer"]

ModelInfo = collections.namedtuple("ModelInfo", ["name", "description"])


def get_available_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            name="eo_arxiv_with_errors",
            description="best overall model, use this for text that might have OCR or spelling errors (default)"
        ),
        ModelInfo(
            name="eo_arxiv_no_errors",
            description="similar to encoder_only_arxiv_with_errors, "
                        "might perform better on text without and worse on text with OCR or spelling errors"
        )
    ]


StringInputOutput = Union[str, List[str]]


class TokenizationRepairer:
    @staticmethod
    def from_pretrained(
            model: str = "eo_arxiv_with_errors",
            use_gpu: bool = True,
            cache_dir: Optional[str] = None
    ) -> "TokenizationRepairer":
        assert any(model == m.name for m in get_available_models()), \
            f"model {model} does not match any of the available models:\n{pprint.pformat(get_available_models())}"

        if cache_dir is None:
            cache_dir = os.environ.get(
                "TOKENIZATION_REPAIR_CACHE_DIR",
                os.path.join(os.path.dirname(__file__), ".cache")
            )

        logger = common.get_logger("TOKENIZATION_REPAIR_FROM_PRETRAINED")
        model_dir = download_model(model, cache_dir, logger)

        return TokenizationRepairer(model_dir, use_gpu)

    @staticmethod
    def from_experiment(
            experiment_dir: str,
            use_gpu: bool = True
    ) -> "TokenizationRepairer":
        return TokenizationRepairer(experiment_dir, use_gpu)

    def __init__(self,
                 model_dir: str,
                 use_gpu: bool) -> None:
        self.logger = common.get_logger("TOKENIZATION_REPAIR")

        if use_gpu and not torch.cuda.is_available():
            self.logger.info("could not find a GPU, using CPU as fallback option")
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        cfg = config.Config.from_yaml(os.path.join(model_dir, "config.yaml"))
        self.logger.debug(f"loaded model config:\n{cfg}")

        self.model = transformer.get_model_from_config(cfg.model, self.device)
        best_checkpoint_path = io.glob_safe(os.path.join(model_dir, "checkpoints", "*-checkpoint-best.pt"))[0]
        best_checkpoint = io.load_checkpoint(best_checkpoint_path)
        self.model.load_state_dict(best_checkpoint["model_state_dict"])
        self.model.eval()

        # set default inference kwargs
        self.inference_kwargs = {
            "temperature": 1.0,
            "temperature_no_spaces": 1.0,
            "thresholds_and_default": None,
            "thresholds_and_default_no_spaces": None
        }

        # check if the corresponding inference pickle files are in the model dir, if so, load them
        if os.path.exists(os.path.join(model_dir, "temperature.pkl")):
            with open(os.path.join(model_dir, "temperature.pkl"), "rb") as tf:
                self.inference_kwargs["temperature"] = pickle.load(tf)
            self.logger.debug(f"Found temperature file: setting temperature to {self.inference_kwargs['temperature']}")
        if os.path.exists(os.path.join(model_dir, "temperature_no_spaces.pkl")):
            with open(os.path.join(model_dir, "temperature_no_spaces.pkl"), "rb") as tf:
                self.inference_kwargs["temperature_no_spaces"] = pickle.load(tf)
            self.logger.debug(f"Found temperature (no spaces) file: setting temperature to "
                             f"{self.inference_kwargs['temperature_no_spaces']}")
        if os.path.exists(os.path.join(model_dir, "thresholds_and_default.pkl")):
            with open(os.path.join(model_dir, "thresholds_and_default.pkl"), "rb") as tf:
                self.inference_kwargs["thresholds_and_default"] = pickle.load(tf)
            self.logger.debug(f"Found thresholds_and_default file: setting thresholds and default to "
                              f"{self.inference_kwargs['thresholds_and_default']}")
        if os.path.exists(os.path.join(model_dir, "thresholds_and_default_no_spaces.pkl")):
            with open(os.path.join(model_dir, "thresholds_and_default_no_spaces.pkl"), "rb") as tf:
                self.inference_kwargs["thresholds_and_default_no_spaces"] = pickle.load(tf)
            self.logger.debug(f"Found thresholds_and_default (no spaces) file: setting thresholds and default to "
                              f"{self.inference_kwargs['thresholds_and_default_no_spaces']}")

        self.max_length = self.model.encoder.config.max_num_embeddings - 2  # - 2 because of bos and eos tokens

    @torch.inference_mode()
    def _repair_text_raw(
            self,
            inputs: List[str],
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> List[inference.InferenceResult]:
        # clean inputs
        inputs = [nlp.clean_sequence(ipt) for ipt in inputs]

        # maybe sort inputs
        input_indices_and_lengths = [(i, len(ipt)) for i, ipt in enumerate(inputs)]
        if sort_by_length:
            input_indices_and_lengths = sorted(input_indices_and_lengths, key=lambda e: e[1], reverse=True)

        inference_results: List[inference.InferenceResult] = [None] * len(inputs)  # type: ignore
        for batch_idx in tqdm(
                range(0, len(inputs), batch_size),
                desc="repairing tokenization",
                leave=False,
                disable=not show_progress
        ):
            batch_input_indices = [idx for idx, _ in input_indices_and_lengths[batch_idx:batch_idx + batch_size]]

            batch = [inputs[idx] for idx in batch_input_indices]
            kwargs = {
                "no_spaces": [" " not in ipt for ipt in batch]
            }
            # add inference keyword arguments to the model
            kwargs.update(self.inference_kwargs)
            batch_inference_results = self.model.inference(
                batch,
                **kwargs
            )
            for idx, ir in zip(batch_input_indices, batch_inference_results):
                inference_results[idx] = ir

        return inference_results

    @torch.inference_mode()
    def repair_text(
            self,
            inputs: StringInputOutput,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> StringInputOutput:
        input_is_string = isinstance(inputs, str)
        assert (
                input_is_string
                or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str))
        ), f"input needs to be a string or a non empty list of strings"

        if input_is_string:
            inputs = [inputs]

        inference_results = self._repair_text_raw(inputs, batch_size, sort_by_length, show_progress)

        outputs = [
            tokenization_repair.repair_whitespace(
                ipt,
                ir.predictions[1:-1]
            )
            for ipt, ir in zip(inputs, inference_results)
        ]
        return outputs[0] if input_is_string else outputs

    @torch.inference_mode()
    def repair_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> Optional[List[str]]:
        inputs = []
        with open(input_file_path, "r", encoding="utf8") as in_file:
            for line in in_file:
                inputs.append(line.strip())

        outputs = self.repair_text(inputs, batch_size, sort_by_length, show_progress)

        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(output)
                    out_file.write("\n")
            return None
        else:
            return outputs
