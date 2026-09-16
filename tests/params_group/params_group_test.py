# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import json
import os

import pytest
import torch
from pydantic import ValidationError

from lm_engine.training.arguments import ModuleParameterMatch, OptimizerArgs, ParamsGroup
from lm_engine.training.distributed import wrap_model_container_for_distributed_training
from lm_engine.training.model_wrapper import get_model_container
from lm_engine.training.optimization.params_group import get_param_groups_list
from lm_engine.training.parallel import ProcessGroupManager

from ..utils import load_training_args_for_unit_tests


_PARAM_GROUPS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "common", "param_groups")


def _load_param_groups(filename: str) -> list[ParamsGroup]:
    # exercise the actual shipped configs, so a regression there is also caught here
    return OptimizerArgs(param_groups=os.path.join(_PARAM_GROUPS_CONFIG_DIR, filename)).param_groups


@pytest.mark.parametrize("use_fsdp", [False, True])
@pytest.mark.parametrize("use_torch_compile", [False, True])
@pytest.mark.parametrize("efficient_initialization", [False, True])
@pytest.mark.parametrize(
    "filename_param_groups",
    [
        ("mup.json", _load_param_groups("mup.yml")),
        ("normal.json", _load_param_groups("normal.yml")),
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


def test_params_group_module_matches() -> None:
    rnn_state_weight_group = ParamsGroup(
        name="rnn_state_weight",
        module_matches=[ModuleParameterMatch(class_name="RNN", parameter_names=["state_weight"])],
    )
    normal_group = ParamsGroup(name="normal", patterns=["*"])
    param_groups = [rnn_state_weight_group, normal_group]

    args = load_training_args_for_unit_tests("params_group/training_config.yml")

    if not ProcessGroupManager.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        ProcessGroupManager()

    model_container = get_model_container(args, efficient_initialization=False, keep_in_fp32=True)
    params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, param_groups)[0]

    result = params_groups.get_param_names()

    assert result["rnn_state_weight"] == ["model.transformer.h.2.sequence_mixer.state_weight"]
    assert "model.transformer.h.3.sequence_mixer.state_weight" in result["normal"]
    assert "model.transformer.h.2.sequence_mixer.state_weight" not in result["normal"]


def test_params_group_module_matches_disambiguates_shared_leaf_class() -> None:
    attention_qkv_group = ParamsGroup(
        name="attention_qkv",
        module_matches=[ModuleParameterMatch(class_name="SoftmaxAttention", parameter_names=["c_attn.weight"])],
    )
    normal_group = ParamsGroup(name="normal", patterns=["*"])
    param_groups = [attention_qkv_group, normal_group]

    args = load_training_args_for_unit_tests("params_group/training_config.yml")

    if not ProcessGroupManager.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        ProcessGroupManager()

    model_container = get_model_container(args, efficient_initialization=False, keep_in_fp32=True)
    params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, param_groups)[0]

    result = params_groups.get_param_names()

    assert result["attention_qkv"] == ["model.transformer.h.0.sequence_mixer.c_attn.weight"]
    assert "model.transformer.h.0.mlp_block.c_fc.weight" in result["normal"]
    assert "model.transformer.h.0.sequence_mixer.c_attn.weight" not in result["normal"]


def test_params_group_requires_module_matches_or_patterns() -> None:
    with pytest.raises(ValidationError):
        ParamsGroup(name="empty")


def test_params_group_lr_multiplier_method() -> None:
    mup_like_group = ParamsGroup(
        name="mup_like",
        patterns=["*.mlp_block.c_fc.weight"],
        params_group_kwargs={"lr_multiplier_method": "1 / m_width"},
    )
    normal_group = ParamsGroup(name="normal", patterns=["*"])
    param_groups = [mup_like_group, normal_group]

    args = load_training_args_for_unit_tests("params_group/training_config.yml")

    if not ProcessGroupManager.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        ProcessGroupManager()

    model_container = get_model_container(args, efficient_initialization=False, keep_in_fp32=True)
    expected_lr = args.optimizer_args.class_args["lr"] / model_container[0].config.m_width

    for _ in range(2):
        params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, param_groups)[0]
        groups_by_name = {group.name: group for group in params_groups.params_groups}

        mup_like_kwargs = groups_by_name["mup_like"].to_param_group()
        assert mup_like_kwargs["lr"] == expected_lr
        assert "lr_multiplier_method" not in mup_like_kwargs

        assert "lr" not in groups_by_name["normal"].to_param_group()


def test_params_group_unknown_lr_multiplier_method_raises() -> None:
    bad_group = ParamsGroup(name="bad", patterns=["*"], params_group_kwargs={"lr_multiplier_method": "does_not_exist"})

    args = load_training_args_for_unit_tests("params_group/training_config.yml")

    if not ProcessGroupManager.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"

        ProcessGroupManager()

    model_container = get_model_container(args, efficient_initialization=False, keep_in_fp32=True)

    with pytest.raises(AssertionError):
        get_param_groups_list(model_container, args.optimizer_args.class_args, [bad_group])
