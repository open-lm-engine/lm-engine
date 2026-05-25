# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from enum import Enum
from typing import Any, Callable

import torch

from .....parallel import ProcessGroupManager
from .communication import AllToAllRotater
from .merge import _Merger, _partial_update


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True


def _is_causal_behavior(rank: int, world_size: int, i: int, causal: bool) -> _CausalBehavior:
    """
    Calculate is_causal behavior for each KV block. The attention can either be
    calculated in full, not at all or with the causal mask applied.
    """
    if not causal:
        return _CausalBehavior.NOT_IS_CAUSAL

    if i == 0:
        return _CausalBehavior.IS_CAUSAL

    source_rank = (rank - i) % world_size
    if source_rank < rank or ProcessGroupManager.get_context_parallel_load_balancing_method() is not None:
        return _CausalBehavior.NOT_IS_CAUSAL
    else:
        return _CausalBehavior.SKIP


def _ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout: float,
    softmax_scale: float | None,
    window_size: int | None,
    softcap: float,
    forward_function: Callable,
) -> tuple[torch.Tensor, ...]:
    if causal and (q.size(1) != k.size(1)):
        raise NotImplementedError("is_causal requires the same query and context sequence lengths")

    if not causal and ProcessGroupManager.get_context_parallel_load_balancing_method() is None:
        raise RuntimeError("Load balancing requires `is_causal=True`.")

    rank = ProcessGroupManager.get_context_parallel_rank()
    world_size = ProcessGroupManager.get_context_parallel_world_size()
    next_kv = None

    # Without making key and value contiguous(), the loss curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    k = k.contiguous()
    v = v.contiguous()

    # Save original shapes so buffer slicing is correct even after chunking.
    k_numel = k.numel()
    k_size = k.size()
    v_size = v.size()

    rotater = AllToAllRotater(1)
    sdpa_merger = _Merger(1)

    for i in range(world_size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            next_kv = rotater.next_buffer()
            k = next_kv[:k_numel].reshape(k_size)
            v = next_kv[k_numel:].reshape(v_size)

        if i < world_size - 1:
            # Send the k, v to the next rank
            next_kv = torch.cat([k.flatten(), v.flatten()])
            rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(rank=rank, world_size=world_size, i=i, causal=causal)

        if is_causal_behavior == _CausalBehavior.SKIP:
            continue

        if i == 0 or (ProcessGroupManager.get_context_parallel_load_balancing_method() is None or not causal):
            # When local balance is enabled, we still need to do SDPA with
            # the both local chunks of q, k, v for the first iteration.
            local_q = q
            local_k = k
            local_v = v
            partial = False
        elif i <= rank:
            # Round-robin load balancing case, and i <= rank.
            # We need to do SDPA with only the first local chunk of k, v.
            # Note that q, k, v each contains two local chunks.
            local_q = q
            local_k = k.chunk(2, dim=1)[0]
            local_v = v.chunk(2, dim=1)[0]
            partial = False
        else:
            # Round-robin load balancing case, and i > rank.
            # We need to do SDPA with only the second half of q, and update
            # only the second part of logsumexp. So partial is True.
            # Note that q, k, v each contains two chunks.
            local_q = q.chunk(2, dim=1)[1]
            local_k = k
            local_v = v
            partial = True

        x, lse, _, _ = forward_function(
            q=local_q,
            k=local_k,
            v=local_v,
            softmax_scale=softmax_scale,
            causal=is_causal_behavior == _CausalBehavior.IS_CAUSAL,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
        )

        sdpa_merger.step(x, lse, partial)

    x, lse = sdpa_merger.results()

    return x, lse


def _ring_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    x: torch.Tensor,
    dx: torch.Tensor,
    lse: torch.Tensor,
    causal: bool,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    rank = ProcessGroupManager.get_context_parallel_rank()
    world_size = ProcessGroupManager.get_context_parallel_world_size()

    next_kv = None
    next_grad_kv = None
    rest: list[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    accum_dtype = torch.float32
    dq = torch.zeros_like(q, dtype=accum_dtype)
    dk = torch.zeros_like(k, dtype=accum_dtype)
    dv = torch.zeros_like(v, dtype=accum_dtype)

    k = k.contiguous()
    v = v.contiguous()

    k_numel = k.numel()
    k_size = k.size()
    v_size = v.size()

    kv_rotater = AllToAllRotater(1)
    dkv_rotater = AllToAllRotater(1)

    for i in range(world_size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            k = buffer[:k_numel].reshape(k_size)
            v = buffer[k_numel:].reshape(v_size)

        if i != world_size - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([k.flatten(), v.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(rank=rank, world_size=world_size, i=i, causal=causal)

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (ProcessGroupManager.get_context_parallel_load_balancing_method() is None or not causal):
                # We need to do SDPA with the full local q, k, v.
                local_q = q
                local_k = k
                local_v = v
                local_x = x
                local_dx = dx
                local_lse = lse
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SDPA with only the first half of k, v.
                # Note that q, k, v each contains two chunks.
                local_q = q
                local_k = k.chunk(2, dim=1)[0]
                local_v = v.chunk(2, dim=1)[0]
                local_x = x
                local_dx = dx
                local_lse = lse
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SDPA with only the second half of q.
                # Note that q, k, v each contains two chunks.
                local_q = q.chunk(2, dim=1)[1]
                local_k = k
                local_v = v
                local_x = x.chunk(2, dim=1)[1]
                local_dx = dx.chunk(2, dim=1)[1]
                local_lse = lse.chunk(2, dim=1)[1].contiguous()

            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            dq_, dk_, dv_, *rest = op(
                query=local_q,
                key=local_k,
                value=local_v,
                out=local_x,
                logsumexp=local_lse,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            dq_ = torch.zeros_like(q, dtype=accum_dtype)
            dk_ = torch.zeros_like(k, dtype=accum_dtype)
            dv_ = torch.zeros_like(v, dtype=accum_dtype)

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            dk += dk_
            dv += dv_
        else:
            # Wait for the kv gradient from (cp_rank - 1) rank.
            next_grad_kv = dkv_rotater.next_buffer()
            dk = next_grad_kv[: dk.numel()].reshape(dk.size())
            dv = next_grad_kv[dk.numel() :].reshape(dv.size())

            if i <= rank and ProcessGroupManager.get_context_parallel_load_balancing_method() is not None:
                dk = _partial_update(dk, dk_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=0, add=True)

                dv = _partial_update(dv, dv_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=0, add=True)
            else:
                dk += dk_
                dv += dv_

        next_grad_kv = torch.cat([dk.flatten(), dv.flatten()])
        # Send the grad key and grad value to the next rank.
        dkv_rotater.exchange_buffers(next_grad_kv)

        if i <= rank or ProcessGroupManager.get_context_parallel_load_balancing_method() is None:
            dq += dq_
        else:
            dq = _partial_update(dq, dq_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=1, add=True)

    assert dk_ is not None
    assert dv_ is not None

    dq = dq.type_as(q)
    next_grad_kv = dkv_rotater.next_buffer().type_as(k)
    dk = next_grad_kv[: dk.numel()].reshape(dk.size())
    dv = next_grad_kv[dk.numel() :].reshape(dv.size())

    return dq, dk, dv, *rest


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        dropout: float,
        softmax_scale: float | None,
        window_size: tuple[int, int],
        softcap: float,
        forward_function: Callable,
        backward_function: Callable,
    ) -> torch.Tensor:
        H = q.size(-1)
        if H % 8 != 0:
            pad = 8 - H % 8
            q = torch.nn.functional.pad(q, [0, pad])
            k = torch.nn.functional.pad(k, [0, pad])
            v = torch.nn.functional.pad(v, [0, pad])

        x, lse = _ring_attention_forward(
            q=q,
            k=k,
            v=v,
            causal=causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
            forward_function=forward_function,
        )

        ctx.save_for_backward(q, k, v, x, lse)
        ctx.dropout = dropout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.backward_function = backward_function

        x = x[..., :H]

        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, x, lse = ctx.saved_tensors

        dq, dk, dv = _ring_attention_backward(
            q=q, k=k, v=v, x=x, dx=dx, lse=lse, causal=ctx.causal, backward_function=ctx.backward_function
        )

        return dq, dk, dv, *[None] * 7


def ring_attention_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout: float,
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
    forward_function: Callable,
    backward_function: Callable,
) -> tuple[torch.Tensor, ...]:
    return _RingAttention.apply(
        q, k, v, causal, dropout, softmax_scale, window_size, softcap, forward_function, backward_function
    )
