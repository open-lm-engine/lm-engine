# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch.nn as nn

from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..hf_models import is_parameter_with_mup_learning_rate, is_parameter_with_no_weight_decay
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


class _ParamsGroupsList(BaseArgs):
    params_groups: list[_ParamsGroup] = []

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
