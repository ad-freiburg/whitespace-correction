import functools
import itertools
from typing import Callable, List, Tuple

import einops

import torch
from torch.nn.utils import rnn


def get_aggregation_fn(aggregation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """

    Return a Pytorch aggregation function (over the first dimension) by name.

    :param aggregation: name of the function
    :return: the function
    """
    if aggregation == "mean":
        return functools.partial(torch.mean, dim=0)
    elif aggregation == "sum":
        return functools.partial(torch.sum, dim=0)
    else:
        raise ValueError(f"aggregation must be mean or sum, but got {aggregation}")


def group_features(
        feats: torch.Tensor,
        groups: List[List[int]],
        agg_fn: Callable[[torch.Tensor], torch.Tensor]
) -> Tuple[torch.Tensor, List[int]]:
    """

    Group features by aggregating the features belonging to the same group.

    :param feats: input features of shape [S, B, H]
    :param groups: grouping information
    :param agg_fn: aggregation function, e.g. torch.mean
    :return: grouped features (padded with zeros to largest number of groups) and group lengths
    """
    assert feats.ndim == 3, f"feats must have a shape of [S, B, H], but got {feats.shape}"
    s, b, h = feats.shape
    group_lengths = [len(group) for group in groups]
    max_group_length = max(group_lengths)
    agg_feats = torch.zeros(max_group_length, b, h, dtype=feats.dtype, device=feats.device)
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        elif len(group) == 1:
            agg_feats[0, i] = agg_fn(feats[:group[0], i])
        start = 0
        end = 1
        while end < len(group):
            if group[start] == group[end]:
                end += 1
                # only continue if we are not at the last group yet
                if end < len(group):
                    continue
            cum_start = sum(group[:start])
            total_group_length = sum(group[start:end])
            agg_feats[start:end, i] = agg_fn(
                einops.rearrange(
                    feats[cum_start:cum_start + total_group_length, i],
                    "(s g) h -> g s h",
                    s=total_group_length // group[start],
                    g=group[start]
                )
            )
            start = end

    return agg_feats, group_lengths
