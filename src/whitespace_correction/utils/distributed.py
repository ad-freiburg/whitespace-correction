from typing import Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


class DistributedInfo:
    def __init__(
            self,
            rank: int,
            local_rank: int,
            world_size: int,
            local_world_size: int
    ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.local_world_size = local_world_size
        assert self.local_rank < torch.cuda.device_count(), \
            f"found {torch.cuda.device_count()} available GPUs, " \
            f"but local_rank is {self.local_rank} (should be in [0..NumGPUs))"
        self.device = torch.device(self.local_rank)

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def __repr__(self) -> str:
        return f"DistributedDevice(rank={self.rank}, local_rank={self.local_rank}, " \
               f"world_size={self.world_size}, local_world_size={self.local_world_size}, device={self.device})"


def unwrap_ddp(ddp: Union[DistributedDataParallel, nn.Module]) -> nn.Module:
    while isinstance(ddp, DistributedDataParallel):
        ddp = ddp.module
    return ddp
