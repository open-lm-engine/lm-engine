# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import logging
from typing import Any

from ..arguments import BaseArgs
from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..logging_utils import log_rank_0
from ..model_wrapper import ModelWrapper
from ..parameter import (
    is_attention_parameter,
    is_parameter_conv_hyperball,
    is_parameter_with_mup_learning_rate,
    is_parameter_with_no_weight_decay,
)


class _ParamsGroup(BaseArgs):
    name: str
    parameter_name_map: dict
    params_group_kwargs: dict = {}

    def to_param_group(self) -> dict:
        result = {}
        result.update(self.params_group_kwargs)

        # do in a sorted order
        param_names = self.get_param_names()

        result["params"] = [self.parameter_name_map[n] for n in param_names]

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
    assert model.config.init_method in (
        "mup",
        "fan_in",
    ), "params_group_method='mup' requires init_method to be 'mup' or 'fan_in'"

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    no_weight_decay_params = {}
    mup_params = {}

    for name, parameter in model.named_parameters():
        if is_parameter_with_mup_learning_rate(parameter):
            mup_params[name] = parameter
        elif is_parameter_with_no_weight_decay(parameter):
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
            _ParamsGroup(
                name="mup",
                parameter_name_map=mup_params,
                params_group_kwargs={"lr": optimizer_class_args["lr"] / model.config.m_width},
            ),
        ]
    )

    return params_group_list


def get_hyperball_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> _ParamsGroupsList:
    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        model = model.model

    conv_hyperball_params = {}
    attention_hyperball_params = {}
    hyperball_params = {}
    no_weight_decay_params = {}
    normal_params = {}

    for name, parameter in model.named_parameters():
        if is_parameter_with_mup_learning_rate(parameter):
            if is_parameter_conv_hyperball(parameter):
                conv_hyperball_params[name] = parameter
            elif is_attention_parameter(parameter):
                attention_hyperball_params[name] = parameter
            else:
                hyperball_params[name] = parameter
        elif is_parameter_with_no_weight_decay(parameter):
            assert not is_parameter_conv_hyperball(parameter)
            assert not is_attention_parameter(parameter)

            no_weight_decay_params[name] = parameter
        else:
            normal_params[name] = parameter

    hybrid_ns = optimizer_class_args.pop("hybrid_ns", False)

    params_group_list = _ParamsGroupsList(
        params_groups=[
            _ParamsGroup(
                name="conv_hyperball",
                parameter_name_map=conv_hyperball_params,
                params_group_kwargs={
                    "hyperball": True,
                    "conv_hyperball_group": True,
                    "weight_decay": 0,
                    "hybrid_ns": hybrid_ns,
                },
            ),
            _ParamsGroup(
                name="attention_hyperball",
                parameter_name_map=attention_hyperball_params,
                params_group_kwargs={
                    "hyperball": True,
                    "attention_hyperball_group": True,
                    "weight_decay": 0,
                    "hybrid_ns": hybrid_ns,
                },
            ),
            _ParamsGroup(
                name="hyperball",
                parameter_name_map=hyperball_params,
                params_group_kwargs={
                    "hyperball": True,
                    "weight_decay": 0,
                    "hybrid_ns": hybrid_ns,
                },
            ),
            _ParamsGroup(
                name="no_weight_decay",
                parameter_name_map=no_weight_decay_params,
                params_group_kwargs={"weight_decay": 0},
            ),
            _ParamsGroup(name="normal", parameter_name_map=normal_params),
        ]
    )

    return params_group_list


_PARAM_GROUPS = {
    None: get_normal_group_with_names,
    ParamsGroupMethod.mup: get_mup_group_with_names,
    ParamsGroupMethod.hyperball: get_hyperball_group_with_names,
}


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None
) -> list[list[_ParamsGroup]]:
    if params_group_method not in _PARAM_GROUPS:
        raise ValueError(f"unexpected `params_group_method` {params_group_method}")

    return [_PARAM_GROUPS[params_group_method](model, optimizer_class_args) for model in model_container]
