# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import logging
from typing import Any

from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..hf_models import is_parameter_with_mup_learning_rate, is_parameter_with_no_weight_decay, get_mup_learning_rate_divisor
from ..model_wrapper import ModelWrapper
from ..utils import BaseArgs, log_rank_0


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


def get_normal_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> _ParamsGroupsList:
    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}

    for name, parameter in model.named_parameters():
        if is_parameter_with_no_weight_decay(parameter):
            no_weight_decay_params[name] = parameter
        else:
            normal_params[name] = parameter

    params_group_list = _ParamsGroupsList(
        params_groups=[
            _ParamsGroup(name="normal", parameter_name_map=normal_params),
            _ParamsGroup(
                name="no_weight_decay",
                parameter_name_map=no_weight_decay_params,
                params_group_kwargs={"weight_decay": 0},
            ),
        ]
    )

    return params_group_list


def get_mup_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> list[_ParamsGroup]:
    assert (
        model.config.init_method == "mup"
    ), "both init method for model and params group method for optimizer should be set to mup"

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}

    # group mup params by their divisor (per-parameter divisor overrides model.config.m_width)
    mup_groups: dict[float, dict[str, Any]] = {}

    for name, parameter in model.named_parameters():
        if is_parameter_with_mup_learning_rate(parameter):
            divisor = get_mup_learning_rate_divisor(parameter) or model.config.m_width
            divisor = float(divisor)
            mup_groups.setdefault(divisor, {})[name] = parameter
        elif is_parameter_with_no_weight_decay(parameter):
            no_weight_decay_params[name] = parameter
        else:
            normal_params[name] = parameter

    params_groups: list[_ParamsGroup] = [
        _ParamsGroup(name="normal", parameter_name_map=normal_params),
        _ParamsGroup(
            name="no_weight_decay",
            parameter_name_map=no_weight_decay_params,
            params_group_kwargs={"weight_decay": 0},
        ),
    ]

    # create one params group per divisor. Keep the original behaviour by naming the
    # default group "mup" when divisor equals model.config.m_width.
    for divisor in sorted(mup_groups.keys()):
        param_map = mup_groups[divisor]
        if divisor == float(model.config.m_width):
            group_name = "mup"
        else:
            # use a stable, inspectable name for non-default divisors
            group_name = f"mup_{str(divisor).replace('.', '_')}"

        params_groups.append(
            _ParamsGroup(
                name=group_name,
                parameter_name_map=param_map,
                params_group_kwargs={"lr": optimizer_class_args["lr"] / divisor},
            )
        )

    return _ParamsGroupsList(params_groups=params_groups)


_PARAM_GROUPS = {
    None: get_normal_group_with_names,
    ParamsGroupMethod.mup: get_mup_group_with_names,
}


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None
) -> list[list[_ParamsGroup]]:
    if params_group_method not in _PARAM_GROUPS:
        raise ValueError(f"unexpected `params_group_method` {params_group_method}")

    return [_PARAM_GROUPS[params_group_method](model, optimizer_class_args) for model in model_container]
