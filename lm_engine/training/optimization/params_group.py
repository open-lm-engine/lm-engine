# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from ..arguments import BaseArgs, ModuleParameterMatch, ParamsGroup
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


def _match_by_module_class(
    model: ModelWrapper, remaining_params: dict, module_matches: list[ModuleParameterMatch]
) -> None:
    matched_params = {}

    for module_name, module in model.named_modules():
        for match in module_matches:
            if type(module).__name__ != match.class_name:
                continue

            for local_name, _ in module.named_parameters(recurse=True):
                if local_name not in match.parameter_names:
                    continue

                full_name = f"{module_name}.{local_name}" if module_name else local_name

                assert full_name in remaining_params
                matched_params[full_name] = remaining_params.pop(full_name)

    return matched_params


def get_param_groups_with_names(
    model: ModelWrapper, optimizer_class_args: dict, param_groups: list[ParamsGroup]
) -> _ParamsGroupsList:
    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    remaining_params = dict(model.named_parameters())
    matched_params_per_group = []

    for group in param_groups:
        if group.module_matches:
            matched_params = _match_by_module_class(model, remaining_params, group.module_matches)
        else:
            matched_params = {}

            for name in list(remaining_params.keys()):
                if any(fnmatch.fnmatch(name, pattern) for pattern in group.patterns):
                    matched_params[name] = remaining_params.pop(name)

        matched_params_per_group.append(matched_params)

    params_groups = []
    for group, matched_params in zip(param_groups, matched_params_per_group):
        params_group_kwargs = dict(group.params_group_kwargs)
        if group.name == "mup" and "lr" not in params_group_kwargs:
            params_group_kwargs["lr"] = optimizer_class_args["lr"] / model.config.m_width

        params_groups.append(
            _ParamsGroup(name=group.name, parameter_name_map=matched_params, params_group_kwargs=params_group_kwargs)
        )

    if remaining_params:
        raise ValueError(
            "the following parameter(s) didn't match any params group's patterns (add a catch-all group, "
            f"e.g. ParamsGroup(name='normal', patterns=['*']), if this is intended): "
            f"{sorted(remaining_params.keys())}"
        )

    result = _ParamsGroupsList(params_groups=params_groups)
    log_rank_0(logging.INFO, f"params groups:\n{result}")

    return result


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, param_groups: list[ParamsGroup]
) -> list[_ParamsGroupsList]:
    return [get_param_groups_with_names(model, optimizer_class_args, param_groups) for model in model_container]
