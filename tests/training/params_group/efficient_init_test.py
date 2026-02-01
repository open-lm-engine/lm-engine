# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import os

import torch

from lm_engine.arguments import UnshardingArgs
from lm_engine.checkpointing import load_checkpoint_and_unshard, save_checkpoint
from lm_engine.distributed import wrap_model_container_for_distributed_training
from lm_engine.hf_models import is_parameter_initialized
from lm_engine.model_wrapper import get_model_container
from lm_engine.utils import ProcessGroupManager, load_yaml, set_seed

from ..test_commons import TestCommons


class EfficientInitTest(TestCommons):
    def test_efficient_init(self) -> None:
        self.skip_test_if_device_unavailable(torch.device("cuda"))

        args = TestCommons.load_training_args_for_unit_tests("params_group/training_config.yml")
        unshard_config = UnshardingArgs(**load_yaml(os.path.join(os.path.dirname(__file__), "unshard.yml")))

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
            args.save_args.save_path = f"tmp-{efficient_initialization}"

            model_container = get_model_container(
                args, efficient_initialization=efficient_initialization, keep_in_fp32=True
            )

            model_container, _ = wrap_model_container_for_distributed_training(args, model_container)
            save_checkpoint(args, model_container, None, None, None, None, 0)

            unshard_config.load_args.load_path = args.save_args.save_path
            unshard_config.load_args.iteration = 0
            unshard_config.unsharded_path = os.path.join(args.save_args.save_path, "unsharded_path")

            _, _, consolidated_state_dict = load_checkpoint_and_unshard(unshard_config)
            models.append(consolidated_state_dict)

        for n in consolidated_state_dict:
            assert is_parameter_initialized(n)
