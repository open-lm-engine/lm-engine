# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from transformers import AutoModelForCausalLM

from lm_engine.hf_models import GPTBaseConfig


config = GPTBaseConfig(
    vocab_size=65024,
    max_position_embeddings=4096,
    hidden_size=8192,
    num_layers=72,
    num_attention_heads=64,
    num_key_value_heads=8,
    intermediate_size=21888,
    position_embedding_type="rope",
    activation_function="swiglu",
    normalization_function="rmsnorm",
    attention_head_type="gqa",
)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

num_parameters = 0
for param_name, param in model.named_parameters():
    num_parameters += param.numel()

print("\ntotal", f"{num_parameters:,}")
