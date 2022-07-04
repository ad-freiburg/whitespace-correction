import glob
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel

from whitespace_correction.utils import common
from whitespace_correction.utils.lr_schedule import LR_SCHEDULER_TYPE

logger = common.get_logger("IO")


def save_checkpoint(checkpoint_path: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    step: int,
                    epoch: int,
                    val_loss: float,
                    lr_scheduler: Optional[LR_SCHEDULER_TYPE] = None,
                    grad_scaler: Optional[amp.GradScaler] = None) -> None:
    """

    Saves a checkpoint to a directory.

    :param checkpoint_path: Filepath to save the checkpoint
    :param model: Pytorch module
    :param optimizer: Pytorch optimizer
    :param step: Global step (to uniquely identify checkpoint)
    :param lr_scheduler: Pytorch learning rate scheduler
    :param grad_scaler: Pytorch grad scaler for mixed precision training
    """

    if isinstance(model, DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    state = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None else lr_scheduler.state_dict(),
        "grad_scaler_state_dict": None if grad_scaler is None else grad_scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss
    }
    torch.save(state, f=checkpoint_path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    """

    Loads a checkpoint from disk. Maps checkpoint values to cpu.

    :param path: Path to the checkpoint file
    :return: Dictionary mapping from string keys to the checkpointed values
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    return checkpoint


def load_state_dict(module: nn.Module,
                    state_dict: Dict[str, Any],
                    prefix: Optional[str] = None) -> None:
    """

    :param module:pytorch module
    :param state_dict: state_dict to load into module
    :param prefix: if not None, only params with this prefix will be loaded (e.g. useful for loading only encoder or
    decoder weights of a model)
    """
    global logger

    current_state_dict = module.state_dict()

    if prefix is not None:
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    current_keys = set(current_state_dict)
    state_dict_keys = set(state_dict)
    if len(current_keys.difference(state_dict_keys)) > 0:
        # raise ValueError
        logger.warning(f"Found keys in module that are not in state_dict: "
                       f"{current_keys.difference(state_dict_keys)}")

    if len(state_dict_keys.difference(current_keys)) > 0:
        logger.warning(f"Found keys in state_dict that are not in module:"
                       f" {state_dict_keys.difference(current_keys)}. Skipping them.")
        state_dict = {k: v for k, v in state_dict.items() if k in current_keys}

    current_state_dict.update(state_dict)

    module.load_state_dict(current_state_dict)


def last_n_checkpoints(checkpoint_dir: str, n: int) -> List[str]:
    """

    Returns the paths to the newest n checkpoints in a checkpoint directory.

    :param checkpoint_dir: path to directory
    :param n: number of checkpoints
    :return: list of newest n checkpoints
    """
    assert os.path.isdir(checkpoint_dir), "checkpoint_dir has to be a directory"
    checkpoints = glob_safe(os.path.join(checkpoint_dir, "*-checkpoint-*.pt"))
    # filter out last and best checkpoints
    checkpoints = [cp for cp in checkpoints
                   if not cp.endswith("checkpoint-last.pt")
                   and not cp.endswith("checkpoint-best.pt")]
    checkpoints = common.natural_sort(checkpoints)
    if n <= 0:
        return checkpoints

    return checkpoints[-n:]


def load_averaged_checkpoint(checkpoint_paths: Union[str, List[str]]) -> Dict[str, Any]:
    """

    Checkpoint averaging.

    :param checkpoint_paths:
    :return:
    """
    if isinstance(checkpoint_paths, str):
        checkpoints = glob_safe(checkpoint_paths)
    else:
        checkpoints = checkpoint_paths
    assert len(checkpoints) >= 1

    state: Dict[str, Any] = {}
    averaged_params = OrderedDict()
    for i, checkpoint_path in enumerate(checkpoints):
        checkpoint = load_checkpoint(checkpoint_path)
        if i == 0:
            state = checkpoint

        model_params = checkpoint["model_state_dict"]

        for k, v in model_params.items():
            if k not in averaged_params:
                averaged_params[k] = v.float().clone()
            else:
                averaged_params[k] += v.float()
    for k in averaged_params.keys():
        averaged_params[k] /= len(checkpoints)

    state["model_state_dict"] = averaged_params
    return state


def glob_safe(pattern: str) -> List[str]:
    """

    Safe version of glob.glob in the sense that it errors when
    no files are found with the glob pattern.

    :param pattern: glob pattern
    :return: files matched by the pattern
    """
    files = glob.glob(pattern.strip())
    if len(files) == 0:
        raise ValueError(f"found no files using glob pattern {pattern}")
    return files


def line_count(filepath: str) -> int:
    """

    Count the number of lines in a file

    :param filepath: path to the file
    :return: number of lines
    """
    with open(filepath, "r") as f:
        for i, _ in enumerate(f):
            pass
    return i + 1
