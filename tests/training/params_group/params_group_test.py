# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os

import pytest
import torch

from lm_engine.distributed import wrap_model_container_for_distributed_training
from lm_engine.enums import ParamsGroupMethod
from lm_engine.hf_models import get_parameter_marker_maps, set_parameter_marker_maps
from lm_engine.model_wrapper import get_model_container
from lm_engine.optimization.params_group import get_param_groups_list
from lm_engine.utils import ProcessGroupManager

from ..utils import load_training_args_for_unit_tests


@pytest.mark.parametrize("use_fsdp", [False, True])
@pytest.mark.parametrize("use_torch_compile", [False, True])
@pytest.mark.parametrize("efficient_initialization", [False, True])
@pytest.mark.parametrize("filename_method", [("mup.json", ParamsGroupMethod.mup), ("normal.json", None)])
def test_params_group(
    use_fsdp: bool,
    use_torch_compile: bool,
    efficient_initialization: bool,
    filename_method: tuple[str, ParamsGroupMethod | None],
) -> None:
    expected_groups_filename, params_group_method = filename_method

    args = load_training_args_for_unit_tests("params_group/training_config.yml")
    args.distributed_args.torch_compile = use_torch_compile
    args.model_args.efficient_initialization = efficient_initialization

    if not ProcessGroupManager.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        ProcessGroupManager()

    model_container = get_model_container(args, efficient_initialization=efficient_initialization, keep_in_fp32=True)

    if use_fsdp:
        model_container, _ = wrap_model_container_for_distributed_training(args, model_container)
    elif use_torch_compile:
        marker_maps = get_parameter_marker_maps(model_container)
        model_container = [torch.compile(model) for model in model_container]
        set_parameter_marker_maps(model_container, marker_maps, _trim_prefix="_orig_mod.")

    params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, params_group_method)[0]

    expected_group = json.load(open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r"))

    tmp = params_groups.get_param_names()

    if use_fsdp or use_torch_compile:
        stripped_resultant_group = {}

        for group_name in tmp:
            stripped_resultant_group[group_name] = [
                param_name.split("_orig_mod.")[-1] for param_name in tmp[group_name]
            ]
    else:
        stripped_resultant_group = tmp

    assert expected_group == stripped_resultant_group
