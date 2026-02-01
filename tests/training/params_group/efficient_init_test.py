# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************


import torch

from lm_engine.distributed import wrap_model_container_for_distributed_training
from lm_engine.hf_models import is_parameter_initialized
from lm_engine.model_wrapper import get_model_container
from lm_engine.utils import ProcessGroupManager, environment

from ..test_commons import TestCommons


class EfficientInitTest(TestCommons):
    def test_efficient_init(self) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")

        if not ProcessGroupManager.is_initialized():
            with environment({"MASTER_ADDR": "localhost", "MASTER_PORT": "29500", "WORLD_SIZE": "1", "RANK": "0"}):
                ProcessGroupManager()

        for efficient_initialization in [False, True]:
            args.model_args.efficient_initialization = efficient_initialization

            model_container = get_model_container(
                args, efficient_initialization=efficient_initialization, keep_in_fp32=True
            )

            model_container, _ = wrap_model_container_for_distributed_training(args, model_container)

            for model in model_container:
                for p in model.parameters():
                    assert is_parameter_initialized(p)
