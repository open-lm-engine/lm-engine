# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.distributed.tensor.experimental._attention import _context_parallel_shard

from ...enums import ContextParallelLoadBalancerMethod
from ..manager import ProcessGroupManager
from .load_balancer import _HeadTailLoadBalancer, _NoLoadBalancer


_LOAD_BALANCERS = {None: _NoLoadBalancer, ContextParallelLoadBalancerMethod.headtail: _HeadTailLoadBalancer}


def prepare_context_parallel_input(
    inputs: tuple[torch.Tensor, ...], input_seq_dim: int = 1
) -> tuple[torch.Tensor, ...]:
    if not ProcessGroupManager.is_context_parallel_enabled():
        return inputs

    cp_mesh = ProcessGroupManager.get_context_parallel_mesh()
    assert isinstance(inputs, tuple)

    S = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)

    load_balancer = _LOAD_BALANCERS[ProcessGroupManager.get_context_parallel_load_balancing_method()](
        S, cp_world_size, cp_mesh.device_type
    )
    inputs = _context_parallel_shard(
        mesh=cp_mesh, buffers=inputs, seq_dims=tuple(input_seq_dim for _ in inputs), load_balancer=load_balancer
    )

    return inputs
