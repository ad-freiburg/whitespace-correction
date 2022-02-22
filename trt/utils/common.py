import logging
import re
from typing import Dict, List, Union

from torch import nn

_LOG_FORMATTER = logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s] %(message)s")


def add_file_log(logger: logging.Logger, log_file: str) -> None:
    """

    Add file logging to an existing logger

    :param logger: logger
    :param log_file: path to logfile
    :return: logger with file logging handler
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_LOG_FORMATTER)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """

    Get a logger that writes to stderr and the specified log file.

    :param name: name of the logger (usually __name__)
    :return: logger
    """

    logger = logging.getLogger(name)
    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(_LOG_FORMATTER)
    if not logger.handlers:
        logger.addHandler(stderr_handler)
    logger.setLevel(logging.INFO)

    return logger


def _eta(dur: float, num_iter: int, total_iter: int) -> float:
    return (dur / num_iter) * total_iter - dur


def eta_minutes(num_minutes: float, num_iter: int, total_iter: int) -> str:
    _eta_minutes = _eta(num_minutes, num_iter, total_iter)
    return f"{num_minutes:.2f} minutes since start, eta: {_eta_minutes:.2f} minutes"


def eta_seconds(num_sec: float, num_iter: int, total_iter: int) -> str:
    _eta_seconds = _eta(num_sec, num_iter, total_iter)
    return f"{num_sec:.2f} seconds since start, eta: {_eta_seconds:.2f} seconds"


def natural_sort(unsorted: List[str], reverse: bool = False) -> List[str]:
    """

    Sort a list of strings naturally (like humans would sort them)
    Example:
        Natural  With normal sorting
        -------  --------
        1.txt    1.txt
        2.txt    10.txt
        10.txt   2.txt
        20.txt   20.txt

    :param unsorted: unsorted list of strings
    :param reverse: reverse order of list
    :return: naturally sorted list
    """

    def _convert(s: str) -> Union[str, int]:
        return int(s) if s.isdigit() else s.lower()

    def _alphanum_key(key: str) -> List[Union[int, str]]:
        return [_convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(unsorted, key=_alphanum_key, reverse=reverse)


def get_num_parameters(module: nn.Module) -> Dict[str, int]:
    """

    Get the number of trainable, fixed and total parameters of a pytorch module.

    :param module: pytorch module
    :return: dict containing number of parameters
    """
    trainable = 0
    fixed = 0
    for p in module.parameters():
        if p.requires_grad:
            trainable += p.numel()
        else:
            fixed += p.numel()
    return {"trainable": trainable, "fixed": fixed, "total": trainable + fixed}
