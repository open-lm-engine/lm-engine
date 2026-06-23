# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import os

from torch.distributed._tensor.api import DTensor
from transformers import AutoModelForCausalLM

from lm_engine.enums import Kernel
from lm_engine.hf_models import GPTBaseConfig
from lm_engine.kernels import enable_kernels
from lm_engine.parallel import ProcessGroupManager

from ....utils import from_config


parser = argparse.ArgumentParser()
parser.add_argument("--activation-function", type=str)
parser.add_argument("--tmp-path", type=str)
args = parser.parse_args()


ProcessGroupManager(tensor_parallel_world_size=int(os.getenv("WORLD_SIZE")))

is_tp_first_rank = ProcessGroupManager.is_tensor_parallel_first_rank()
num_key_value_heads = 8

config = GPTBaseConfig(
    num_layers=2,
    position_embedding_type="learned_absolute",
    vocab_size=50257,
    max_position_embeddings=512,
    hidden_size=128,
    normalization_function="layernorm",
    initializer_range=0.02,
    use_cache=True,
    bos_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    rope_theta=10000,
    rope_scaling=None,
    rope_dim=None,
    m_emb=None,
    m_width=None,
    m_residual=None,
    init_method="normal",
    embedding_init_method="normal",
    use_depth_scaled_init=False,
    router_aux_loss_coef=0.001,
    tie_word_embeddings=False,
    sequence_mixer_blocks={
        "sequence_mixer_type": "softmax_attention",
        "add_bias": False,
        "num_attention_heads": 16,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": None,
        "softmax_dropout": 0,
        "dropout": 0,
        "attention_multiplier": None,
        "attention_multiplier_method": "1 / sqrt(head_dim)",
        "attention_gate": False,
        "exclusive_self_attention": False,
        "sliding_window": None,
    },
    mlp_blocks=[
        {"mlp_type": "MLP", "activation_function": args.activation_function, "add_bias": False},
        {
            "mlp_type": "MoE",
            "activation_function": args.activation_function,
            "add_bias": False,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "shared_intermediate_size": None,
        },
    ],
)
enable_kernels([Kernel.scattermoe]).__enter__()


if is_tp_first_rank:
    with ProcessGroupManager.set_dummy_tensor_parallel_world_size(1):
        model = from_config(config)

    model.save_pretrained(args.tmp_path, safe_serialization=True)

ProcessGroupManager.barrier()

model_tp = AutoModelForCausalLM.from_pretrained(args.tmp_path)
tp_state_dict = model_tp.state_dict()

cpu_state_dict = {key: value.to("cpu") for key, value in tp_state_dict.items()}

tp_state_dict_unsharded = {
    key: value.full_tensor() if isinstance(value, DTensor) else value for key, value in cpu_state_dict.items()
}

ProcessGroupManager.barrier()

if is_tp_first_rank:
    original_state_dict = model.state_dict()

    assert tp_state_dict_unsharded.keys() == original_state_dict.keys()
    for key in original_state_dict:
        assert original_state_dict[key].equal(tp_state_dict_unsharded[key])
