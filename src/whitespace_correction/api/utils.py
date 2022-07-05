import logging
import os
import platform
import re
import shutil
import zipfile
from typing import List, Optional, Tuple, Union

import requests

import tokenizers

import torch
from torch import nn

from tqdm import tqdm

from whitespace_correction.utils import common, constants, tables
from whitespace_correction.utils.inference import Beam, ScoreFn, log_likelihood_score_fn

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
            "EMNLP_tokenization_repair_transformer_BHW_2022.materials"
_NAME_TO_URL = {
    "eo_large_arxiv_with_errors": f"{_BASE_URL}/eo_large_arxiv_with_errors.zip",
    "eo_small_arxiv_with_errors": f"{_BASE_URL}/eo_small_arxiv_with_errors.zip",
    "eo_medium_arxiv_with_errors": f"{_BASE_URL}/eo_medium_arxiv_with_errors.zip",
    "nmt_large_arxiv_with_errors": f"{_BASE_URL}/nmt_large_arxiv_with_errors.zip",
    "nmt_medium_arxiv_with_errors": f"{_BASE_URL}/nmt_medium_arxiv_with_errors.zip",
    "nmt_small_arxiv_with_errors": f"{_BASE_URL}/nmt_small_arxiv_with_errors.zip",
}


def _unpack_zip(
        zip_file_path: str,
        directory: str
) -> None:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(directory)


def get_download_dir() -> str:
    return os.environ.get(
        "WHITESPACE_CORRECTION_DOWNLOAD_DIR",
        os.path.join(os.path.dirname(__file__), ".download")
    )


def get_cache_dir() -> str:
    return os.environ.get(
        "WHITESPACE_CORRECTION_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), ".cache")
    )


def get_tensorrt_cache_dir() -> str:
    return os.environ.get(
        "WHITESPACE_CORRECTION_TENSORRT_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), ".tensorrt_cache")
    )


def download_model(
        name: str,
        download_dir: Optional[str],
        cache_dir: Optional[str],
        force_download: bool,
        logger: logging.Logger
) -> str:
    """

    Downloads and extracts a model into cache dir and returns the path to the model directory

    :param name: unique name of the model
    :param download_dir: directory to store zipped model
    :param cache_dir: directory to cache unzipped model
    :param force_download: download model even if it is already in the cache dir
    :param logger: instance of a logger to log some useful information
    :return: path of the model directory
    """
    if name not in _NAME_TO_URL:
        raise RuntimeError(f"no URL for model {name}, should not happen")
    url = _NAME_TO_URL[name]

    download_dir = download_dir or get_download_dir()

    zip_file_path = os.path.join(download_dir, url.split("/")[-1])
    model_zip_not_downloaded = not os.path.exists(zip_file_path)
    if model_zip_not_downloaded or force_download:
        directory = os.path.dirname(zip_file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        logger.info(f"downloading model {name} from {_NAME_TO_URL[name]} to download directory {download_dir}")
        response = requests.get(url, stream=True)
        if not response.ok:
            raise RuntimeError(f"error downloading the model {name} from {_NAME_TO_URL[name]}: "
                               f"status {response.status_code}, {response.reason}")

        try:
            file_size = int(response.headers.get("content-length", 0))
            pbar = tqdm(
                desc=f"Downloading model {name}",
                total=file_size,
                ascii=True,
                leave=False,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024
            )

            with open(zip_file_path, "wb") as of:
                for data in response.iter_content():
                    of.write(data)
                    pbar.update(len(data))

            pbar.close()

        except BaseException as e:
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            raise e
    else:
        logger.info(f"model {name} is already downloaded to download directory {download_dir}")

    cache_dir = cache_dir or get_cache_dir()

    model_dir = os.path.join(cache_dir, name)
    model_not_extracted = not os.path.exists(model_dir)
    if model_not_extracted or force_download:
        shutil.rmtree(model_dir, ignore_errors=True)
        _unpack_zip(zip_file_path, model_dir)

    experiment_dir = os.listdir(model_dir)
    assert len(experiment_dir) == 1, f"zip file for model {name} should contain exactly one subdirectory, " \
                                     f"but found {len(experiment_dir)}"
    return os.path.join(model_dir, experiment_dir[0])


def get_cpu_info() -> str:
    if platform.system() == "Linux":
        try:
            cpu_regex = re.compile(r"model name\t: (.*)", re.DOTALL)
            with open("/proc/cpuinfo", "r", encoding="utf8") as inf:
                cpu_info = inf.readlines()

            for line in cpu_info:
                line = line.strip()
                match = cpu_regex.match(line)
                if match is not None:
                    return match.group(1)
        except BaseException:
            return platform.processor()
    return platform.processor()


def get_gpu_info(device: Union[torch.device, str, int]) -> str:
    device_props = torch.cuda.get_device_properties(device)
    return f"{device_props.name} ({device_props.total_memory // 1024 // 1024:,}MiB memory, " \
           f"{device_props.major}.{device_props.minor} compute capability, " \
           f"{device_props.multi_processor_count} multiprocessors)"


def get_device_info(device: torch.device) -> str:
    return get_gpu_info(device) if device.type == "cuda" else get_cpu_info()


def split(items: List, lengths: List[int]) -> List[List]:
    assert sum(lengths) == len(items)
    split_items = []
    offset = 0
    for length in lengths:
        split_items.append(items[offset:offset + length])
        offset += length
    return split_items


def sliding_windows(length: int, window_size: int) -> List[int]:
    return list(range(0, length, window_size))


def match_token_ids_ignoring_space_and_unk(
        token_ids: List[int],
        tokenizer: tokenizers.Tokenizer,
        left_context: str,
        window: str,
        right_context: str
) -> Tuple[int, int]:
    left_context_pattern = r"\s*".join(
        re.escape(char) if tokenizer.token_to_id(char) else "."
        for char in left_context.replace(" ", "")
    )
    right_context_pattern = r"\s*".join(
        re.escape(char) if tokenizer.token_to_id(char) else "."
        for char in right_context.replace(" ", "")
    )
    window_pattern = r"\s*".join(
        re.escape(char) if tokenizer.token_to_id(char) else "."
        for char in window.replace(" ", "")
    )
    if window.startswith(" "):
        window_pattern = r"(\s*" + window_pattern
    else:
        window_pattern = r"\s*(" + window_pattern
    if window.endswith(" "):
        window_pattern = window_pattern + r"\s*)"
    else:
        window_pattern = window_pattern + r")\s*"

    unk_token_id = tokenizer.token_to_id(constants.UNK)
    search_str = "".join(
        tokenizer.id_to_token(token_id) if token_id != unk_token_id else "#"
        for token_id in token_ids
    )
    pattern = re.compile(left_context_pattern + window_pattern + right_context_pattern)
    match = pattern.search(search_str)
    assert match is not None, f"could no match the following two strings:" \
                              f"\n{left_context + window + right_context}\n{search_str}"
    return match.start(1), match.end(1)


def generate_report(
        task: str,
        model_name: str,
        model: nn.Module,
        inputs: List[str],
        runtime: float,
        precision: torch.dtype,
        batch_size: int,
        sort_by_length: bool,
        device: torch.device,
        file_path: Optional[str] = None
) -> Optional[str]:
    input_size = len(inputs)
    input_size_chars = sum(len(ipt) for ipt in inputs)

    if precision == torch.float16:
        precision_str = "fp16"
    elif precision == torch.bfloat16:
        precision_str = "bfp16"
    elif precision == torch.float32:
        precision_str = "fp32"
    else:
        raise ValueError("expected precision to be one of torch.float16, torch.bfloat16 or torch.float32")

    report = tables.generate_table(
        header=[
            "Task",
            "Model",
            "Input size",
            "Runtime in seconds",
            "Seq/s",
            "kChar/s",
            "MiB GPU memory",
            "Mio. parameters",
            "Precision",
            "Batch size",
            "Sorted",
            "Device"
        ],
        data=[
            [
                task,
                model_name,
                f"{input_size:,} sequences, {input_size_chars:,} chars",
                f"{runtime:.1f}",
                f"{input_size / runtime:.1f}",
                f"{input_size_chars / (runtime * 1000):.1f}",
                f"{torch.cuda.max_memory_reserved(device) // (1024 ** 2):,}" if device.type == "cuda" else "-",
                f"{common.get_num_parameters(model)['total'] / (1000 ** 2):,.1f}",
                precision_str,
                str(batch_size),
                "yes" if sort_by_length else "no",
                f"{torch.cuda.get_device_name(device)}, {get_cpu_info()}" if device.type == "cuda" else get_cpu_info()
            ]
        ],
        fmt="markdown"
    )
    if file_path is not None:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        exists = os.path.exists(file_path)
        with open(file_path, "a" if exists else "w", encoding="utf8") as of:
            if exists:
                of.write(report.splitlines()[-1] + "\n")
            else:
                of.write(report + "\n")
        return None
    else:
        return report


def char2char_score_fn(char_tokenizer: tokenizers.Tokenizer) -> ScoreFn:
    bos_token_id = char_tokenizer.token_to_id(constants.BOS)
    eos_token_id = char_tokenizer.token_to_id(constants.EOS)
    unk_token_id = char_tokenizer.token_to_id(constants.UNK)
    ws_token_id = char_tokenizer.token_to_id(" ")

    log_l_score = log_likelihood_score_fn()

    def _score(beam: Beam, input_str_no_spaces: Optional[str] = None) -> float:
        assert input_str_no_spaces is not None
        assert beam.token_ids[0] == bos_token_id and len(beam.token_ids) > 1

        pred_token_id = beam.token_ids[-1]
        token_ids = beam.token_ids[1:-1]
        prev_pred_token_id = token_ids[-1] if len(token_ids) > 0 else unk_token_id

        input_str_no_spaces_position = sum(token_id != ws_token_id for token_id in token_ids)

        s = log_l_score(beam, None)

        # only allow whitespaces (but not successively), input characters or eos
        if input_str_no_spaces_position >= len(input_str_no_spaces):
            # must be eos
            if pred_token_id != eos_token_id:
                s -= 1_000_000
        else:
            input_token_id = char_tokenizer.token_to_id(
                input_str_no_spaces[input_str_no_spaces_position]
            ) or unk_token_id
            if (
                    # do not allow:
                    # 1. two whitespaces in a row
                    (pred_token_id == ws_token_id and prev_pred_token_id == ws_token_id)
                    # 2. predicting something else than whitespace or input char
                    or (pred_token_id != ws_token_id and pred_token_id != input_token_id)
                    # 3. too early eos
                    or pred_token_id == eos_token_id
            ):
                s -= 1_000_000

        return s

    return _score
