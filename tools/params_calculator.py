# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

"""Calculate the total and active parameter count for a given training config, without
downloading a tokenizer or materializing any real weights (the model is built on the meta
device, mirroring what ModelWrapper.calculate_num_parameters does internally).

Usage:
    python tools/params_calculator.py <path-to-training-config.yml>
"""

import argparse

from lm_engine.training.arguments import TrainingArgs
from lm_engine.training.kernels import enable_kernels
from lm_engine.training.model_wrapper.base import ModelWrapper
from lm_engine.training.utils import load_yaml


def calculate_num_parameters(config_path: str) -> dict[str, int]:
    args = TrainingArgs(**load_yaml(config_path))

    # bypass ModelWrapper.__init__ (which also downloads a tokenizer and builds the real model)
    # and only set the handful of attributes that _setup_config/_get_model_kwargs/
    # calculate_num_parameters actually need
    model_wrapper = ModelWrapper.__new__(ModelWrapper)
    model_wrapper.model_name = args.model_args.model_name
    model_wrapper.pretrained_config = args.model_args.pretrained_config
    model_wrapper.trust_remote_code = args.model_args.trust_remote_code
    model_wrapper.use_padding_free_transformer = args.model_args.use_padding_free_transformer
    model_wrapper.sequence_parallel = args.distributed_args.sequence_parallel
    model_wrapper.num_pipeline_stages = args.distributed_args.num_pipeline_stages
    model_wrapper.pipeline_stage_id = 0
    model_wrapper.is_pipeline_parallel_enabled = model_wrapper.num_pipeline_stages > 1

    with enable_kernels(args.kernel_args.kernels):
        model_wrapper._setup_config()
        return model_wrapper.calculate_num_parameters(return_dict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="calculate the number of parameters for a training config")
    parser.add_argument("--config", help="path to the training config yaml file")
    args = parser.parse_args()

    result = calculate_num_parameters(args.config)

    print(f"total parameters = {result['num_parameters']:,}")
    print(f"active parameters = {result['active_parameters']:,}")


if __name__ == "__main__":
    main()
