# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

from transformers import set_seed

from lm_engine.distributed import wrap_model_container_for_distributed_training
from lm_engine.model_wrapper import get_model_container
from lm_engine.utils import ProcessGroupManager

from ..test_commons import TestCommons


class EfficientInitTest(TestCommons):
    def test_efficient_init(self) -> None:
        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")

        if not ProcessGroupManager.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"

            ProcessGroupManager()

        models = []

        for efficient_initialization in [False, True]:
            set_seed(args.random_args.seed)
            args.model_args.efficient_initialization = efficient_initialization

            model_container = get_model_container(
                args, efficient_initialization=efficient_initialization, keep_in_fp32=False
            )

            model_container, _ = wrap_model_container_for_distributed_training(args, model_container)
            models.append(model_container)

        for n, p in models[0].named_parameters():
            assert models[1].state_dict()[n] == p
