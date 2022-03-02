import io
import logging
import os
import platform
import re
import shutil
import zipfile
from typing import List

import requests
import torch
from tqdm import tqdm

_NAME_TO_URL = {
    "eo_arxiv_with_errors": "https://tokenization.cs.uni-freiburg.de/transformer/eo_arxiv_with_errors.zip",
    "eo_small_arxiv_with_errors": "https://tokenization.cs.uni-freiburg.de/transformer/eo_small_arxiv_with_errors.zip",
}


def download_model(name: str, cache_dir: str, force_download: bool, logger: logging.Logger) -> str:
    """

    Downloads and extracts a model into cache dir and returns the path to the model directory

    :param name: unique name of the model
    :param cache_dir: directory to store the model
    :param force_download: download model even if it is already in the cache dir
    :param logger: instance of a logger to log some useful information
    :return: path of the model directory
    """

    model_dir = os.path.join(cache_dir, name)
    model_does_not_exist = not os.path.exists(model_dir)
    if model_does_not_exist or force_download:
        if name not in _NAME_TO_URL:
            raise RuntimeError(f"no URL for model {name}, should not happen")

        logger.info(f"downloading model {name} from {_NAME_TO_URL[name]} to cache directory {cache_dir}")
        response = requests.get(_NAME_TO_URL[name], stream=True)
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

            buf = io.BytesIO()
            for data in response.iter_content():
                buf.write(data)
                pbar.update(len(data))

            with zipfile.ZipFile(buf, "r") as zip_file:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir)
                zip_file.extractall(model_dir)

            pbar.close()

        except Exception as e:
            # only remove the model dir on error when it did not exist before
            if model_does_not_exist:
                shutil.rmtree(model_dir)
            raise e
    else:
        logger.info(f"model {name} was already downloaded to cache directory {cache_dir}")

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
        except Exception:
            return platform.processor()
    return platform.processor()


def get_gpu_info() -> str:
    device_props = torch.cuda.get_device_properties("cuda")
    return f"{device_props.name} ({device_props.total_memory // 1024 // 1024:,}MiB memory, " \
           f"{device_props.major}.{device_props.minor} compute capability, " \
           f"{device_props.multi_processor_count} multiprocessors)"


def split(items: List, lengths: List[int]) -> List[List]:
    assert sum(lengths) == len(items)
    split_items = []
    offset = 0
    for length in lengths:
        split_items.append(items[offset:offset + length])
        offset += length
    return split_items
