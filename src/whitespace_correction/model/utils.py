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
        return torch.mean
    elif aggregation == "sum":
        return torch.sum
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
    assert feats.dim == 3, "feats must have a shape of [S, B, H]"
    agg_feats = []
    for feat, group in zip(einops.rearrange(feats, "s b h -> b s h"), groups):
        agg_feat = []
        for grouped_feats in torch.split(feat[:sum(group)], group):
            agg_feat.append(agg_fn(grouped_feats))
        agg_feats.append(torch.stack(agg_feat))
    return rnn.pad_sequence(agg_feats), [len(group) for group in groups]
