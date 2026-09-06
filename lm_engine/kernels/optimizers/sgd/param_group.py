# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from dataclasses import dataclass, field

import torch


@dataclass
class SGDParamsGroup:
    params: dict[str, torch.Tensor]
    lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool
    step: int = 0
    momentum_buffers: dict[str, torch.Tensor | None] = field(init=False)
    _lazy_init: bool = False

    def __post_init__(self) -> None:
        self.momentum_buffers = {}

        if self.momentum == 0 or self._lazy_init:
            return

        for name, W in self.params.items():
            self.momentum_buffers[name] = torch.zeros_like(
                W, dtype=torch.float32, memory_format=torch.contiguous_format
            )

    def get_params_for_optimization(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor | None]]:
        params = []
        grads = []
        momentum_buffer_list = []
        for name, W in self.params.items():
            if W.grad is None:
                continue

            params.append(W)
            grads.append(W.grad)

            # lazy initialization for momentum buffers
            if self._lazy_init and self.momentum != 0 and self.momentum_buffers.get(name) is None:
                self.momentum_buffers[name] = torch.zeros_like(
                    W, dtype=torch.float32, memory_format=torch.contiguous_format
                )

            momentum_buffer_list.append(self.momentum_buffers.get(name))

        return params, grads, momentum_buffer_list

    def increment_step(self) -> int:
        self.step += 1
        return self.step
