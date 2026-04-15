# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer


class SplitParamOptimizer(Optimizer):
    def __init__(
        self,
        inner: Optimizer,
        proxy_grad_fns: dict[int, tuple[nn.Parameter, Callable]],
        split_params: set[nn.Parameter],
    ) -> SplitParamOptimizer:
        self._inner = inner
        self._proxy_grad_fns = proxy_grad_fns
        self._split_params = split_params

    @property
    def param_groups(self) -> list[dict]:
        return self._inner.param_groups

    @property
    def state(self) -> dict:
        return self._inner.state

    def state_dict(self) -> dict:
        return self._inner.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._inner.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict) -> None:
        self._inner.add_param_group(param_group)

    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        for group in self._inner.param_groups:
            for p in group["params"]:
                info = self._proxy_grad_fns.get(id(p))
                if info is None:
                    continue

                orig_param, grad_slice_fn = info
                if orig_param.grad is not None:
                    p.grad = grad_slice_fn(orig_param.grad)

        return self._inner.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._inner.zero_grad(set_to_none)

        for param in self._split_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def __repr__(self) -> str:
        x = super().__repr__()
        return f"{x}({self._inner})"
