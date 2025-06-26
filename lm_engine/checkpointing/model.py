# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from ..containers import ModelContainer


class _ModelSaver(Stateful):
    def __init__(self, model_container: ModelContainer) -> None:
        self.model_container = model_container

    def state_dict(self) -> dict:
        state_dict = {}

        for model in self.model_container:
            model_state_dict = get_model_state_dict(model)
            if model.has_teacher_model():
                model_state_dict = self._filter_out_teacher_state_dict(model_state_dict)

            state_dict.update(model_state_dict)

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        for model in self.model_container:
            set_model_state_dict(
                model, model_state_dict=state_dict, options=StateDictOptions(strict=not model.has_teacher_model())
            )

    def _filter_out_teacher_state_dict(self, state_dict: dict) -> dict:
        result = {}
        for key, value in state_dict.items():
            if not "teacher_model" in key:
                result[key] = value

        return result


def _get_model_path(path: str) -> str:
    return os.path.join(path, "model")
