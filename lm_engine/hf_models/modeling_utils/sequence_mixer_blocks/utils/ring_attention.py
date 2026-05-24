# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import Any

import torch
from torch.distributed import ProcessGroup

from .....parallel import ProcessGroupManager
from .communication import _Merger
from .flash_attention_utils import _get_flash_attention_function


def _ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout: float,
    softmax_scale: float | None,
    sliding_window: int | None,
    softcap: float,
    group: ProcessGroup,
) -> tuple[torch.Tensor, ...]:
    _flash_attention_function, _flash_attention_varlen_function = _get_flash_attention_function(dropout=dropout)

    if causal and (q.size(1) != k.size(1)):
        raise NotImplementedError("is_causal requires the same query and context sequence lengths")

    if not causal and ProcessGroupManager.get_load_balancing_method() is not None:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    next_kv = None

    # Without making key and value contiguous(), the loss curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    k = k.contiguous()
    v = v.contiguous()

    sdpa_merger = _Merger(seq_dim=1)
    rest: list[Any]
    out: torch.Tensor
    logsumexp: torch.Tensor

    rotater = _create_rotater(group, 2)

    for i in range(world_size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            k = next_kv[: k.numel()].reshape(k.size())
            v = next_kv[k.numel() :].reshape(v.size())

        if i < world_size - 1:
            # Send the k, v to the next rank
            next_kv = torch.cat([k.flatten(), v.flatten()])
            next_kv = rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(rank=rank, world_size=size, i=i, is_causal=is_causal)

        # For a detailed understanding of the load balancing algorithm, see
        # Note [Context parallelism load balance algorithm for causal masking]
        if is_causal_behavior == _CausalBehavior.SKIP:
            # If i > rank and load balancing is not turned on.
            continue

        if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            q, k, v, partial = (q, k, v, False)
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SDPA with only the first local chunk of k, v.
            # Note that q, k, v each contains two local chunks.
            ROUND_ROBIN_CYCLE = 2
            q, k, v, partial = (
                q,
                k.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                v.chunk(ROUND_ROBIN_CYCLE, dim=2)[0],
                False,
            )
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SDPA with only the second half of q, and update
            # only the second part of logsumexp. So partial is True.
            # Note that q, k, v each contains two chunks.
            q, k, v, partial = q.chunk(2, dim=2)[1], k, v, True

        # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
        # for the SDPA kernel definitions.
        out, logsumexp, *rest = _flash_attention_function(
            q,
            k,
            v,
            is_causal=is_causal_behavior.value,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )

        sdpa_merger.step(out, logsumexp, partial)

    # pyrefly: ignore [unbound-name]
    return *sdpa_merger.results(), *rest
