# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed

from lm_engine.arguments import TrainingArgs, UnshardingArgs
from lm_engine.checkpointing import ensure_last_checkpoint_is_saved, load_checkpoint_for_inference, save_checkpoint
from lm_engine.distributed import wrap_model_container_for_distributed_training
from lm_engine.enums import Mode
from lm_engine.model_wrapper import get_model_container
from lm_engine.utils import ProcessGroupManager, load_yaml


parser = argparse.ArgumentParser()
parser.add_argument("--train-config", type=str)
parser.add_argument("--unshard-config", type=str)
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tmp-path", type=str)
parser.add_argument("--zero-stage", type=int)
parser.add_argument("--data-parallel-replication-world-size", type=int)
parser.add_argument("--data-parallel-sharding-world-size", type=int)
args = parser.parse_args()

train_config = TrainingArgs(**load_yaml(args.train_config))
unshard_config = UnshardingArgs(**load_yaml(args.unshard_config))

if args.attention_head_type == "mha":
    num_key_value_heads = train_config.model_args.pretrained_config["sequence_mixer_blocks"][0]["num_attention_heads"]
elif args.attention_head_type == "mqa":
    num_key_value_heads = 1
else:
    num_key_value_heads = 8

# set zero stage
train_config.distributed_args.stage = args.zero_stage
# attention head type
for block in train_config.model_args.pretrained_config["sequence_mixer_blocks"]:
    block["num_key_value_heads"] = num_key_value_heads
# activation function
for block in train_config.model_args.pretrained_config["mlp_blocks"]:
    block["activation_function"] = args.activation_function

ProcessGroupManager(
    tensor_parallel_world_size=train_config.distributed_args.tensor_parallel_world_size,
    pipeline_parallel_world_size=train_config.distributed_args.pipeline_parallel_world_size,
)

global_rank = ProcessGroupManager.get_global_rank()

if global_rank == 0:
    with (
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
        ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
    ):
        original_num_stages = train_config.distributed_args.num_pipeline_stages
        train_config.distributed_args.num_pipeline_stages = 1

        model_container = get_model_container(train_config, Mode.training)
        model_container[0].save_pretrained(os.path.join(args.tmp_path, "single_rank"))

        train_config.distributed_args.num_pipeline_stages = original_num_stages

torch.distributed.barrier()

# modify args to load the saved single_rank checkpoint
train_config.model_args.pretrained_config = None
train_config.model_args.model_name = os.path.join(args.tmp_path, "single_rank")
train_config.save_args.save_path = os.path.join(args.tmp_path, "save")

# modify unsharding args to load the checkpoint for unsharding
iteration = 0
unshard_config.load_args.load_path = train_config.save_args.save_path
unshard_config.load_args.iteration = iteration
unshard_config.unsharded_path = os.path.join(args.tmp_path, "unsharded_path")

parallel_model_container = get_model_container(train_config, Mode.training)
parallel_model_container, _ = wrap_model_container_for_distributed_training(train_config, parallel_model_container)

save_checkpoint(
    train_config,
    model_container=parallel_model_container,
    optimizer_container=None,
    lr_scheduler_container=None,
    train_dataloader=None,
    experiments_tracker=None,
    iteration=iteration,
    metadata=None,
)

ensure_last_checkpoint_is_saved()

torch.distributed.barrier()

_, _, consolidated_state_dict = load_checkpoint_for_inference(
    unshard_config, mode=Mode.unsharding, allowed_meta_device=False
)

if global_rank == 0:
    original_state_dict = model_container[0].state_dict()

    assert consolidated_state_dict.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(consolidated_state_dict[key])
