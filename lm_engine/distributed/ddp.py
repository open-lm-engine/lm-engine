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
        sync_module_states: bool = False,
        overlap_communication: bool = False,
    ) -> DDP:
        super().__init__()

        self.model = model
        self.device_mesh = ProcessGroupManager.get_data_parallel_mesh() if device_mesh is None else device_mesh
        self.process_group = self.device_mesh.get_group()
        self.overlap_communication = overlap_communication

        if self.overlap_communication:
            raise NotImplementedError()

        if sync_module_states:
            torch.distributed._broadcast_coalesced(
                process_group=self.process_group,
                tensors=list(self.parameters()) + list(self.buffers()),
                buffer_size=250 * (1024**2),
                src=0,
            )

        for parameter in self.parameters():
            if parameter.requires_grad:
                parameter.register_hook(self._all_reduce_hook)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def _all_reduce_hook(self, grad: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(grad, op=ReduceOp.AVG, group=self.process_group)
        return grad
