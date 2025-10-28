# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os

import torch
from parameterized import parameterized

from lm_engine.distributed import (
    _get_parameter_marker_maps,
    _set_parameter_marker_maps,
    wrap_model_container_for_distributed_training,
)
from lm_engine.enums import ParamsGroupMethod
from lm_engine.model_wrapper import get_model_container
from lm_engine.optimization.params_group import get_param_groups_list
from lm_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class ParamsGroupTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix(
            [False, True], [False, True], [False, True], [("mup.json", ParamsGroupMethod.mup), ("normal.json", None)]
        )
    )
    def test_mup_group(
        self,
        use_fsdp: bool,
        use_torch_compile: bool,
        efficient_initialization: bool,
        filename_method: tuple[str, ParamsGroupMethod | None],
    ) -> None:
        expected_groups_filename, params_group_method = filename_method

        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")
        args.distributed_args.torch_compile = use_torch_compile
        args.model_args.efficient_initialization = efficient_initialization

        if not ProcessGroupManager.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"

            ProcessGroupManager()

        model_container = get_model_container(
            args, efficient_initialization=efficient_initialization, keep_in_fp32=True
        )

        if use_fsdp:
            model_container, _ = wrap_model_container_for_distributed_training(args, model_container)
        elif use_torch_compile:
            marker_maps = _get_parameter_marker_maps(model_container)
            model_container = [torch.compile(model) for model in model_container]
            _set_parameter_marker_maps(model_container, marker_maps)

        params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, params_group_method)[0]

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )

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
