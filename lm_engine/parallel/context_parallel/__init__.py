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

from .load_balancer import _HeadTailLoadBalancer


_LOAD_BALANCERS = {"headtail": _HeadTailLoadBalancer}


def prepare_context_parallel_input(
    inputs: tuple[torch.Tensor, ...], cp_mesh: DeviceMesh, load_balancer_type: str | None = "headtail"
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
    inputs, _ = cp_shard(cp_mesh=cp_mesh, inputs=inputs, load_balancer_type=load_balancer_type)

    return inputs


def cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    load_balancer_type: str | None = "headtail",
    input_seq_dim: int = 1,
) -> tuple[torch.Tensor, ...]:
    """
    Shard inputs and attention masks across the context parallel mesh.

    This function distributes input tensors across devices in the CP mesh
    along the sequence dimension, enabling efficient processing. It optionally
    uses a load balancer to handle uneven computation workload.

    Args:
        cp_mesh: Device mesh for context parallel dimension
        inputs: Tuple of input tensors to be sharded along the sequence
            dimension
        load_balancer_type: Type of load balancer to use. Options:
            - "headtail": Use HeadTailLoadBalancer (for SDPA)
            - "ptrr": Use PTRRLoadBalancer (for FlexAttention)
            - None: Disable load balancing
            Defaults to "headtail".
        input_seq_dim: Sequence dimension index for sharding. Defaults to 1,
            which covers most use cases where tensors have shape
            [batch_size, seq_len]. Can be changed by passing a
            different value if your tensors use a different sequence
            dimension layout.

    Returns:
        - sharded_inputs: Tuple of input tensors sharded along the
            sequence dimension

    Raises:
        ValueError: If load_balancer_type is "ptrr" and attention_masks
            is None or a dict
    """

    assert isinstance(inputs, tuple)

    S = inputs[0].size(input_seq_dim)
    cp_world_size = cp_mesh.size(0)

    load_balancer = _LOAD_BALANCERS[load_balancer_type](S, cp_world_size, cp_mesh.device_type)

    inputs = _context_parallel_shard(
        mesh=cp_mesh, buffers=inputs, seq_dims=tuple(input_seq_dim for _ in inputs), load_balancer=load_balancer
    )

    return inputs


def _context_parallel_shard(
    mesh: DeviceMesh,
    buffers: CPBufferContainer,
    seq_dims: CPBufferSeqDims,
    load_balancer: _LoadBalancer | None = None,
) -> list[torch.Tensor | BlockMask]:
    """
    Shard the buffers along the specified sequence dimensions (`seq_dims`), so that each
    rank retains only its corresponding shard according to the provided `mesh`. If a
    `load_balancer` is provided, the buffers will be rearranged by the load balancer
    before sharding to improve load balance. Buffers can be either tensors or `BlockMask`
    objects. If a buffer is a `BlockMask`, its sharding dimension is determined by the
    `BlockMask` implementation, and the corresponding `seq_dim` is ignored.

    Note:
        For `_context_parallel_shard`, a non-None `load_balancer` must be explicitly passed
        if load balancing is required.

    Args:
        mesh (DeviceMesh): The device mesh used for context parallelism.
        buffers (List[torch.Tensor | BlockMask]): Buffers whose usage depends on the sequence
            dimension. Examples include input batches, labels, and positional embedding buffers.
            These buffers must be sharded along the sequence dimension to ensure correctness.
        seq_dims (List[int]): The sequence dimensions for each buffer in `buffers`. Must have
            the same length as `buffers`.
        load_balancer (Optional[_LoadBalancer]): An optional load balancer object. If provided,
            it rearranges the buffers before sharding to achieve better load balance. If not
            provided, no rearrangement is performed.

    Returns:
        List[torch.Tensor | BlockMask]: The sharded buffers, each corresponding to the local
            shard for the current rank.
    """
    # TODO: these global variables are going to bite us someday.
    # We will have to remove them soon.
    # For the new API, we only support the module wrapper mode.
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MODULE_WRAPPER
    global _cp_options
    if load_balancer is not None:
        _cp_options.enable_load_balance = True
    else:
        _cp_options.enable_load_balance = False

    if len(buffers) != len(seq_dims):
        raise ValueError("`seq_dims` must have the same number of elements as `buffers`.")

    flat_buffers, spec = tree_flatten(buffers)
    flat_seq_dims, _ = tree_flatten(seq_dims)
    if len(flat_buffers) != len(flat_seq_dims):
        raise ValueError("`seq_dims` must have the pytree structure as `buffers`.")

    if isinstance(flat_buffers[0], torch.Tensor):
        device = flat_buffers[0].device
    else:
        device = flat_buffers[0].kv_num_blocks.device
    for buffer in flat_buffers:
        if isinstance(buffer, torch.Tensor):
            assert device == buffer.device, "All buffers must be on the same device"
        else:
            assert device == buffer.kv_num_blocks.device, "All buffers must be on the same device"

    flat_sharded_buffers = _context_parallel_buffers(mesh, flat_buffers, flat_seq_dims, load_balancer)

    return tree_unflatten(flat_sharded_buffers, spec)
