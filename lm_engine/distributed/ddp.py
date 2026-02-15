# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup, ReduceOp


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
            with torch.no_grad():
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
        return self._model(*args, **kwargs)

    def state_dict(self) -> dict:
        return self._model.state_dict()

    def _all_reduce_hook(self, grad: torch.Tensor) -> torch.Tensor:
        torch.distributed.all_reduce(grad, op=ReduceOp.AVG, group=self.process_group)
        return grad

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        return self._model.named_parameters(prefix, recurse, remove_duplicate)

    def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
        return self._model.named_buffers()

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        return self._model.named_modules()

    def named_children(self):
        return self._model.named_children()
