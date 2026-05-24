# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.experimental._attention import _context_parallel_shard

from .load_balancer import _HeadTailLoadBalancer


_LOAD_BALANCERS = {"headtail": _HeadTailLoadBalancer}


def prepare_context_parallel_input(
    inputs: tuple[torch.Tensor, ...],
    cp_mesh: DeviceMesh,
    input_seq_dim: int = 1,
    load_balancer_type: str | None = "headtail",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Shard inputs, labels, positions, and attention masks for Context Parallel.

    The caller must provide ``extra_kwargs["positions"]`` before calling this
    function.  Position resolution (per-document vs sequential) is handled
    upstream in ``post_dataloading_process``.

    Args:
        inputs: Input tensor of shape [batch_size, seq_len]
        labels: Label tensor of shape [batch_size, seq_len]
        extra_kwargs: Dictionary containing 'positions' (required) and
            optionally 'attention_masks' to be sharded.
        cp_mesh: Device mesh for context parallel dimension
        device: Device for the tensors
        load_balancer_type: Type of load balancer to use for sharding.
            Options: "headtail", "ptrr", or None. Defaults to "headtail".

    Returns:
        Tuple of (sharded_inputs, sharded_labels, updated_extra_kwargs) where:
            - sharded_inputs: Inputs sharded along sequence dimension
            - sharded_labels: Labels sharded along sequence dimension
            - updated_extra_kwargs: Dict with sharded 'positions' and optionally
              sharded 'attention_masks'
    """

    assert isinstance(inputs, tuple)

    S = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)

    load_balancer = _LOAD_BALANCERS[load_balancer_type](S, cp_world_size, cp_mesh.device_type)
    inputs = _context_parallel_shard(
        mesh=cp_mesh, buffers=inputs, seq_dims=tuple(input_seq_dim for _ in inputs), load_balancer=load_balancer
    )

    return inputs
