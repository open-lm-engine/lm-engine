# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup, ReduceOp

from ..utils import Accelerator


class DDP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        process_group: ProcessGroup,
        sync_module_states: bool = False,
        overlap_communication: bool = False,
    ) -> DDP:
        super().__init__()

        self._model = model
        self.overlap_communication = overlap_communication
        self.process_group = process_group

        if self.overlap_communication:
            raise NotImplementedError()

        if sync_module_states:
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    with torch.device(Accelerator.get_current_device()):
                        module.reset_parameters()

            with torch.no_grad():
                torch.distributed._broadcast_coalesced(
                    process_group=self.process_group,
                    tensors=list(self._model.parameters()) + list(self._model.buffers()),
                    buffer_size=250 * (1024**2),
                    src=0,
                )

        for parameter in self._model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(self._all_reduce_hook)

    def forward(self, *args, **kwargs) -> Any:
        return self._model(*args, **kwargs)

    def _all_reduce_hook(self, grad: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(grad, op=ReduceOp.AVG, group=self.process_group)
        return grad

    def process_marker_map(self, marker_map: dict) -> dict:
        new_marker_map = {}
        for key, value in marker_map.items():
            new_marker_map[f"_model.{key}"] = value

        return new_marker_map

    def __setattr__(self, name, value):
        return self._model.__setattr__(name, value)

    def __getattribute__(self, name):
        return super().__getattribute__(name)
