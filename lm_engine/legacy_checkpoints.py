# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch

from .modeling_utils.mlp_blocks.mlp.utils import interleave_up_gate_tensor_for_mlp
from .models import GPTBaseForCausalLM


CONFIG_BACKFILL = {
    "embedding_init_method": "normal",
    "use_depth_scaled_init": False,
    "tie_word_embeddings": True,
}

_LEGACY_INTERLEAVE_KEYS = ("use_interleaved_weights", "use_interleaved_weights_for_shared_experts")


def is_legacy_checkpoint(config_dict: dict) -> bool:
    return any(key not in config_dict for key in CONFIG_BACKFILL)


def backfill_config(config_dict: dict) -> None:
    for key, value in CONFIG_BACKFILL.items():
        config_dict.setdefault(key, value)


def pop_legacy_interleave_flags(config_dict: dict) -> list[tuple[bool, bool]]:
    return [
        tuple(mlp_block.pop(key, False) for key in _LEGACY_INTERLEAVE_KEYS)
        for mlp_block in config_dict.get("mlp_blocks", [])
    ]


def _interleave(tensor: torch.Tensor, dim: int) -> None:
    u, g = tensor.chunk(2, dim=dim)
    tensor.copy_(interleave_up_gate_tensor_for_mlp(u, g, dim=dim))


@torch.no_grad()
def interleave_legacy_glu_weights(model: GPTBaseForCausalLM, interleave_flags: list[tuple[bool, bool]]) -> None:
    for block, (experts_interleaved, shared_interleaved) in zip(
        model.transformer.h.values(), interleave_flags, strict=True
    ):
        mlp = block.mlp_block

        if not getattr(mlp, "is_glu", False):
            continue

        for linear, interleaved in (
            (mlp.c_fc, experts_interleaved),
            (getattr(mlp, "c_fc_shared", None), shared_interleaved),
        ):
            if linear is None or interleaved:
                continue
            dim = linear.weight.ndim - 2
            _interleave(linear.weight, dim=dim)
            if linear.bias is not None:
                _interleave(linear.bias, dim=dim)
