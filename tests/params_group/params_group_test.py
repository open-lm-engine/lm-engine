# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import json
import os

import pytest
import torch

from lm_engine.training.arguments import ParamsGroup
from lm_engine.training.distributed import wrap_model_container_for_distributed_training
from lm_engine.training.model_wrapper import get_model_container
from lm_engine.training.optimization.params_group import get_param_groups_list
from lm_engine.training.parallel import ProcessGroupManager

from ..utils import load_training_args_for_unit_tests


_MUP_GROUP = ParamsGroup(
    name="mup",
    patterns=[
        "*.mlp_block.c_fc.weight",
        "*.mlp_block.c_proj.weight",
        "*.mlp_block.c_fc_shared.weight",
        "*.mlp_block.c_proj_shared.weight",
        "*.mlp_block.gate.weight",
        "*.mlp_block.up_fc.weight",
        "*.mlp_block.gate_fc.weight",
        "*.sequence_mixer.c_attn.weight",
        "*.sequence_mixer.c_proj.weight",
        "*.sequence_mixer.conv1d.weight",
        "*.sequence_mixer.in_proj.weight",
        "*.sequence_mixer.out_proj.weight",
        "*.sequence_mixer.input_projection.weight",
        "*.sequence_mixer.output_projection.weight",
        "*.sequence_mixer.state_weight",
        "*.sequence_mixer.D",
        "*.sequence_mixer.decay_gate.A_log",
    ],
)

_NO_WEIGHT_DECAY_GROUP = ParamsGroup(
    name="no_weight_decay",
    patterns=[
        "*.ln_1.weight",
        "*.ln_2.weight",
        "*.ln_f.weight",
        "*.norm.weight",
        "*.bias",
        "*.dt_bias",
        "*.state_weight",
        "*.D",
        "*.decay_gate.A_log",
    ],
    params_group_kwargs={"weight_decay": 0},
)

# catch-all: get_param_groups_with_names raises if any parameter matches no group's patterns
_NORMAL_GROUP = ParamsGroup(name="normal", patterns=["*"])


@pytest.mark.parametrize("use_fsdp", [False, True])
@pytest.mark.parametrize("use_torch_compile", [False, True])
@pytest.mark.parametrize("efficient_initialization", [False, True])
@pytest.mark.parametrize(
    "filename_param_groups",
    [
        ("mup.json", [_MUP_GROUP, _NO_WEIGHT_DECAY_GROUP, _NORMAL_GROUP]),
        ("normal.json", [_NO_WEIGHT_DECAY_GROUP, _NORMAL_GROUP]),
    ],
)
def test_params_group(
    use_fsdp: bool,
    use_torch_compile: bool,
    efficient_initialization: bool,
    filename_param_groups: tuple[str, list[ParamsGroup]],
) -> None:
    expected_groups_filename, param_groups = filename_param_groups

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
        model_container = [torch.compile(model) for model in model_container]

    params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, param_groups)[0]

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
