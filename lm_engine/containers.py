# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch.nn as nn


class _Container:
    def __init__(self, model_list: list[nn.Module]) -> _Container:
        self.model_list = model_list

    def __iter__(self):
        for model in self.model_list:
            yield model

    def __getitem__(self, index: int) -> nn.Module:
        return self.model_list[index]

    def __setitem__(self, index: int, model: nn.Module) -> None:
        self.model_list[index] = model

    def __len__(self) -> int:
        return len(self.model_list)

    def __str__(self) -> str:
        return str(self.model_list)


class ModelContainer(_Container):
    def train(self) -> ModelContainer:
        for model in self:
            model.train()

    def eval(self) -> ModelContainer:
        for model in self:
            model.eval()

        return self


class LRSchedulerContainer(_Container):
    def step(self) -> None:
        for lr_scheduler in self:
            lr_scheduler.step()


class OptimizerContainer(LRSchedulerContainer):
    def zero_grad(self) -> None:
        for optimizer in self:
            optimizer.zero_grad()


class BackwardHookOptimizerContainer(OptimizerContainer):
    def step(self) -> None:
        return

    def zero_grad(self) -> None:
        return
