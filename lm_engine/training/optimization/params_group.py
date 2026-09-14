# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from ..arguments import BaseArgs, ParamsGroup
from ..containers import ModelContainer
from ..logging_utils import log_rank_0
from ..model_wrapper import ModelWrapper


class _ParamsGroup(BaseArgs):
    name: str
    parameter_name_map: dict
    params_group_kwargs: dict = {}

    def to_param_group(self) -> dict:
        result = {}
        result.update(self.params_group_kwargs)

        # do in a sorted order
        param_names = self.get_param_names()

        result["params"] = []
        for param_name in param_names:
            result["params"].append(self.parameter_name_map[param_name])

        return result

    def get_param_names(self) -> list[str]:
        param_names = list(self.parameter_name_map.keys())
        param_names.sort()
        return param_names

    def __len__(self) -> int:
        return len(self.parameter_name_map)

    def __str__(self) -> str:
        lines = [f"{self.name} ({len(self)} params):"]
        lines.extend(f"    {param_name}" for param_name in self.get_param_names())
        return "\n".join(lines)

    __repr__ = __str__


class _ParamsGroupsList(BaseArgs):
    params_groups: list[_ParamsGroup] = []

    def model_post_init(self, __context: Any) -> None:
        self.params_groups = list(filter(lambda group: len(group) > 0, self.params_groups))
        super().model_post_init(__context)

    def add_params_group(self, params_group: _ParamsGroup) -> None:
        self.params_groups.append(params_group)

    def to_torch_compatible_params_groups(self) -> list[dict]:
        return [group.to_param_group() for group in self.params_groups]

    def get_param_names(self) -> list[str]:
        return {group.name: group.get_param_names() for group in self.params_groups}

    def __str__(self) -> str:
        return "\n".join(str(group) for group in self.params_groups)

    __repr__ = __str__


def get_param_groups_with_names(
    model: ModelWrapper, optimizer_class_args: dict, param_groups: list[ParamsGroup]
) -> _ParamsGroupsList:
    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    remaining_params = dict(model.named_parameters())
    params_groups = []

    for group in param_groups:
        matched_params = {}
        for name in list(remaining_params.keys()):
            if any(fnmatch.fnmatch(name, pattern) for pattern in group.patterns):
                matched_params[name] = remaining_params.pop(name)

        params_group_kwargs = dict(group.params_group_kwargs)
        if group.name == "mup" and "lr" not in params_group_kwargs:
            params_group_kwargs["lr"] = optimizer_class_args["lr"] / model.config.m_width

        params_groups.append(
            _ParamsGroup(name=group.name, parameter_name_map=matched_params, params_group_kwargs=params_group_kwargs)
        )

    params_groups.append(_ParamsGroup(name="normal", parameter_name_map=remaining_params))

    result = _ParamsGroupsList(params_groups=params_groups)
    log_rank_0(logging.INFO, f"params groups:\n{result}")

    return result


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, param_groups: list[ParamsGroup]
) -> list[_ParamsGroupsList]:
    return [get_param_groups_with_names(model, optimizer_class_args, param_groups) for model in model_container]
