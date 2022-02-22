import io
import logging
import os
import zipfile

import requests

_NAME_TO_URL = {
    "eo_arxiv_with_errors": "https://github.com/bastiscode/trt/tokenization_repair/zip_files/eo_arxiv_with_errors.zip",
    "eo_arxiv_no_errors": "https://github.com/bastiscode/trt/tokenization_repair/zip_files/eo_arxiv_no_errors.zip"
}


def download_model(name: str, cache_dir: str, logger: logging.Logger) -> str:
    """

    Downloads and extracts a model into cache dir and returns the path to the model directory

    :param name: unique name of the model
    :param cache_dir: directory to store the model
    :param logger: instance of a logger to log some useful information
    :return: path of the model directory
    """

    model_dir = os.path.join(cache_dir, name)
    if os.path.exists(model_dir):
        logger.info(f"model {name} was already downloaded to cache directory {cache_dir}")
        return model_dir

    if name not in _NAME_TO_URL:
        raise RuntimeError(f"no URL for model {name}, should not happen")

    logger.info(f"downloading model {name} to cache directory {cache_dir}")
    request = requests.get(_NAME_TO_URL[name])
    if not request.ok:
        raise RuntimeError(f"error downloading the model {name}: status={request.status_code}, {request.reason}")

    try:
        os.makedirs(model_dir)
        with zipfile.ZipFile(io.BytesIO(request.content)) as zip_file:
            zip_file.extractall(model_dir)
    except Exception as e:
        os.removedirs(model_dir)
        raise RuntimeError(f"error extracting downloaded zipfile: {e}")

    experiment_dir = os.listdir(model_dir)
    assert len(experiment_dir) == 1, f"zip file for model {name} should contain exactly one subdirectory, " \
                                     f"but found {len(experiment_dir)}"
    return os.path.join(model_dir, experiment_dir[0])
