# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..containers import ModelContainer, OptimizerContainer


class _OptimizerSaver(Stateful):
    def __init__(self, model_container: ModelContainer, optimizer_container: OptimizerContainer) -> None:
        self.model_container = model_container
        self.optimizer_container = optimizer_container

    def state_dict(self) -> dict:
        state_dict = {}

        for model, optimizer in zip(self.model_container, self.optimizer_container):
            optimizer_state_dict = get_optimizer_state_dict(
                model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
            )
            state_dict.update(optimizer_state_dict)

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        for model, optimizer in zip(self.model_container, self.optimizer_container):
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )


def _get_optimizer_path(path: str) -> str:
    return os.path.join(path, "optimizer")
