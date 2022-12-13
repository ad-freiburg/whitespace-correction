import logging
import os
import platform
import re
import shutil
import zipfile
from typing import List, Optional, Tuple, Union

import requests

import torch
from torch import nn

from tqdm import tqdm

from whitespace_correction.model.tokenizer import Tokenizer
from whitespace_correction.utils import common, constants, tables

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
            "EMNLP_whitespace_correction_transformer_BHW_2022.materials"
_NAME_TO_URL = {
    "eo_large": f"{_BASE_URL}/eo_large.zip",
    "eo_small": f"{_BASE_URL}/eo_small.zip",
    "eo_medium": f"{_BASE_URL}/eo_medium.zip",
    "ed_large": f"{_BASE_URL}/ed_large.zip",
    "ed_medium": f"{_BASE_URL}/ed_medium.zip",
    "ed_small": f"{_BASE_URL}/ed_small.zip",
}


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
