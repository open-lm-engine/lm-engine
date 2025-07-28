# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed
from torch.distributed._tensor.api import DTensor

from lm_engine.dtensors import dtensor_to_tensor
from lm_engine.enums import Kernel
from lm_engine.hf_models import (
    GPTBaseConfig,
    fix_unsharded_state_dict,
    get_model_parallel_class,
    unshard_tensor_parallel_state_dicts,
)
from lm_engine.kernels import enable_kernels
from lm_engine.utils import ProcessGroupManager

from ...test_common import TestCommons


parser = argparse.ArgumentParser()
parser.add_argument("--attention-head-type", type=str)
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


ProcessGroupManager(tensor_parallel_world_size=int(os.getenv("WORLD_SIZE")))

is_tp_first_rank = ProcessGroupManager.is_tensor_parallel_first_rank()

if args.attention_head_type == "mha":
    num_key_value_heads = 16
elif args.attention_head_type == "mqa":
    num_key_value_heads = 1
else:
    num_key_value_heads = 8

config = GPTBaseConfig(
    num_layers=2,
    position_embedding_type="learned_absolute",
    hidden_size=128,
    sequence_mixer_blocks=[
        {
            "sequence_mixer_type": "softmax_attention",
            "add_bias": False,
            "num_attention_heads": 16,
            "num_key_value_heads": num_key_value_heads,
        },
        {
            "sequence_mixer_type": "softmax_attention",
            "add_bias": False,
            "num_attention_heads": 16,
            "num_key_value_heads": num_key_value_heads,
        },
    ],
    mlp_blocks=[
        {"mlp_type": "MLP", "activation_function": args.activation_function, "add_bias": False},
        {"mlp_type": "MoE", "activation_function": args.activation_function, "add_bias": False},
    ],
)
enable_kernels([Kernel.scattermoe]).__enter__()


if is_tp_first_rank:
    model = TestCommons.from_config(None, config)
    model.save_pretrained(args.tmp_path, safe_serialization=True)

torch.distributed.barrier()

model_tp = get_model_parallel_class(config.model_type).from_pretrained(args.tmp_path)
tp_state_dict = model_tp.state_dict()


def run_check(fix: bool):
    cpu_state_dict = {key: value.to("cpu") for key, value in tp_state_dict.items()}

    if fix:
        tp_state_dict_unsharded = {
            key: value.full_tensor() if isinstance(value, DTensor) else value for key, value in cpu_state_dict.items()
        }
        tp_state_dict_unsharded = fix_unsharded_state_dict(
            config, tp_state_dict_unsharded, ProcessGroupManager.get_tensor_parallel_world_size()
        )
    else:
        cpu_state_dict = {key: dtensor_to_tensor(value) for key, value in cpu_state_dict.items()}
        torch.save(
            cpu_state_dict, os.path.join(args.tmp_path, f"tp-{ProcessGroupManager.get_tensor_parallel_rank()}.pt")
        )
        del cpu_state_dict

        torch.distributed.barrier()

        tensor_parallel_state_dicts = [
            torch.load(os.path.join(args.tmp_path, f"tp-{i}.pt"), weights_only=False)
            for i in range(ProcessGroupManager.get_tensor_parallel_world_size())
        ]

        tp_state_dict_unsharded = unshard_tensor_parallel_state_dicts(
            config, tensor_parallel_state_dicts=tensor_parallel_state_dicts
        )

    torch.distributed.barrier()

    if is_tp_first_rank:
        original_state_dict = model.state_dict()

        assert tp_state_dict_unsharded.keys() == original_state_dict.keys()
        for key in original_state_dict:
            assert original_state_dict[key].equal(tp_state_dict_unsharded[key])


run_check(True)
run_check(False)
