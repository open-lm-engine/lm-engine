# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import json
import os

import torch
from parameterized import parameterized

from dolomite_engine.distributed import _get_parameter_marker_maps, _set_parameter_marker_maps
from dolomite_engine.enums import Mode, ParamsGroupMethod
from dolomite_engine.model_wrapper import get_model_container
from dolomite_engine.optimization.params_group import get_param_groups_list
from dolomite_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class ParamsGroupTest(TestCommons):
    @parameterized.expand(
        TestCommons.make_args_matrix([False, True], [("mup.json", ParamsGroupMethod.mup), ("normal.json", None)])
    )
    def test_mup_group(self, use_fsdp: bool, filename_method: tuple[str, ParamsGroupMethod | None]) -> None:
        expected_groups_filename, params_group_method = filename_method
        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")

        with (
            torch.device("meta"),
            ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
            ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
            ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
            ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ):
            model_container = get_model_container(args, Mode.training)

            if use_fsdp:
                marker_maps = _get_parameter_marker_maps(model_container)
                model_container = [torch.compile(model) for model in model_container]
                _set_parameter_marker_maps(model_container, marker_maps)

        params_groups = get_param_groups_list(model_container, args.optimizer_args.class_args, params_group_method)[0]

        expected_group = json.load(
            open(os.path.join(os.path.dirname(__file__), "groups", expected_groups_filename), "r")
        )

        stripped_resultant_group = params_groups.get_param_names()

        if use_fsdp:
            tmp = stripped_resultant_group
            stripped_resultant_group = {}

            for group_name in tmp:
                stripped_resultant_group[group_name] = [
                    param_name.split("_orig_mod.")[-1] for param_name in tmp[group_name]
                ]

        assert expected_group == stripped_resultant_group
