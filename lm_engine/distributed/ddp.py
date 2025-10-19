# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh, ReduceOp

from ..utils import ProcessGroupManager


class DDP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh | None = None,
        sync_parameters: bool = False,
        sync_buffers: bool = False,
        overlap_all_reduce: bool = False,
    ) -> DDP:
        super().__init__()

        self.model = model
        self.device_mesh = ProcessGroupManager.get_data_parallel_mesh() if device_mesh is None else device_mesh
        self.process_group = self.device_mesh.get_group()
        self.overlap_all_reduce = overlap_all_reduce

        if self.overlap_all_reduce:
            raise NotImplementedError()

        if sync_parameters:
            for parameter in self.parameters():
                torch.distributed.broadcast(parameter, src=0, group=self.process_group)

        if sync_buffers:
            for parameter in self.buffers():
                torch.distributed.broadcast(parameter, src=0, group=self.process_group)

        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.register_hook(self._all_reduce_hook)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def _all_reduce_hook(self, grad: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(grad, op=ReduceOp.AVG, group=self.process_group)
        return grad
