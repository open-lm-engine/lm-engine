# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from ..containers import ModelContainer, OptimizerContainer


class _ModelOptimizerSaver(Stateful):
    def __init__(self, model_container: ModelContainer, optimizer_container: OptimizerContainer) -> None:
        self.model_container = model_container
        self.optimizer_container = optimizer_container

    def _filter_out_teacher_state_dict(self, state_dict: dict) -> dict:
        result = {}
        for key, value in state_dict.items():
            if not "teacher_model" in key:
                result[key] = value

        return result

    def _model_state_dict(self) -> dict:
        state_dict = {}

        for model in self.model_container:
            model_state_dict = get_model_state_dict(model)
            if model.has_teacher_model():
                model_state_dict = self._filter_out_teacher_state_dict(model_state_dict)

            state_dict.update(model_state_dict)

        return state_dict

    def _optimizer_state_dict(self) -> dict:
        state_dict = {}

        for model, optimizer in zip(self.model_container, self.optimizer_container):
            optimizer_state_dict = get_optimizer_state_dict(
                model, optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
            )
            state_dict.update(optimizer_state_dict)

        return state_dict

    def _load_model_state_dict(self, state_dict: dict) -> None:
        for model in self.model_container:
            set_model_state_dict(
                model, model_state_dict=state_dict, options=StateDictOptions(strict=not model.has_teacher_model())
            )

    def _load_optimizer_state_dict(self, state_dict: dict) -> None:
        for model, optimizer in zip(self.model_container, self.optimizer_container):
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )

    def state_dict(self) -> dict:
        return {
            "model": self._model_state_dict(),
            "optimizer": self._optimizer_state_dict() if self.optimizer_container is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._load_model_state_dict(state_dict["model"])
        if self.optimizer_container is not None:
            self._load_optimizer_state_dict(state_dict["optimizer"])


def _get_model_optimizer_path(path: str) -> str:
    return os.path.join(path, "model_optimizer")
