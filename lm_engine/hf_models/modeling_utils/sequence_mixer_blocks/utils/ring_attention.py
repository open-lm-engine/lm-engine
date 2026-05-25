# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from enum import Enum
from typing import Any

import torch

from .....parallel import ProcessGroupManager
from .communication import AllToAllRotater
from .merge import _Merger
from .utils import _get_flash_attention_function


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
) -> tuple[torch.Tensor, ...]:
    _, _, _flash_attention_forward, _flash_attention_backward = _get_flash_attention_function(dropout=dropout)

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

    rotater = AllToAllRotater(seq_dim=1)
    sdpa_merger = _Merger(seq_dim=1)
    out: torch.Tensor
    logsumexp: torch.Tensor

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

        out, logsumexp, _, _ = _flash_attention_forward(
            q=local_q,
            k=local_k,
            v=local_v,
            softmax_scale=softmax_scale,
            causal=is_causal_behavior == _CausalBehavior.IS_CAUSAL,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
        )

        sdpa_merger.step(out, logsumexp, partial)

    return sdpa_merger.results()


def _ring_attention_backward(
    grad_out: torch.Tensor,
    grad_out_name: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
    rank: int,
    world_size: int,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    next_kv = None
    next_grad_kv = None
    rest: list[Any]
    grad_query_, grad_key_, grad_value_ = None, None, None

    accum_dtype = torch.float32
    grad_query = torch.zeros_like(query, dtype=accum_dtype)
    grad_key = torch.zeros_like(key, dtype=accum_dtype)
    grad_value = torch.zeros_like(value, dtype=accum_dtype)

    key = key.contiguous()
    value = value.contiguous()
    kv_rotater = _create_rotater(group, 2)
    dkv_rotater = _create_rotater(group, 2, method=_RotateMethod.ALL_TO_ALL)
    for i in range(size):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            pointer = 0
            key = buffer[pointer : pointer + key.numel()].reshape(key.shape)
            pointer += key.numel()
            value = buffer[pointer : pointer + value.numel()].reshape(value.shape)
            pointer += value.numel()

        if i != size - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([key.flatten(), value.flatten()])
            kv_rotater.exchange_buffers(next_kv)

        is_causal_behavior = _is_causal_behavior(rank=rank, world_size=size, i=i, is_causal=is_causal)

        if is_causal_behavior != _CausalBehavior.SKIP:
            if i == 0 or (not _cp_options.enable_load_balance or not is_causal):
                # We need to do SDPA with the full local q, k, v.
                q, k, v, out_, dout, lse = (query, key, value, out, grad_out, logsumexp)
            elif i <= rank:
                # Round-robin load balancing case, and i <= rank.
                # We need to do SDPA with only the first half of k, v.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query,
                    key.chunk(2, dim=seq_dim)[0],
                    value.chunk(2, dim=seq_dim)[0],
                    out,
                    grad_out,
                    logsumexp,
                )
            else:
                # Round-robin load balancing case, and i > rank.
                # We need to do SDPA with only the second half of q.
                # Note that q, k, v each contains two chunks.
                q, k, v, out_, dout, lse = (
                    query.chunk(2, dim=seq_dim)[1],
                    key,
                    value,
                    out.chunk(2, dim=seq_dim)[1],
                    grad_out.chunk(2, dim=seq_dim)[1],
                    # Need to make logsumexp contiguous, otherwise there will
                    # be numerical error.
                    logsumexp.chunk(2, dim=seq_dim)[1].contiguous(),
                )

            kwargs[grad_out_name] = dout
            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            grad_query_, grad_key_, grad_value_, *rest = op(
                query=q,
                key=k,
                value=v,
                out=out_,
                logsumexp=lse,
                is_causal=is_causal_behavior.value,
                **kwargs,
            )
        else:
            grad_query_ = torch.zeros_like(query, dtype=accum_dtype)
            grad_key_ = torch.zeros_like(key, dtype=accum_dtype)
            grad_value_ = torch.zeros_like(value, dtype=accum_dtype)

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            grad_key += grad_key_
            grad_value += grad_value_
        else:
            pointer = 0
            # Wait for the kv gradient from (cp_rank - 1) rank.
            next_grad_kv = dkv_rotater.next_buffer()
            grad_key = next_grad_kv[pointer : pointer + grad_key.numel()].reshape(grad_key.shape)
            pointer += grad_key.numel()
            grad_value = next_grad_kv[pointer : pointer + grad_value.numel()].reshape(grad_value.shape)

            if i <= rank and _cp_options.enable_load_balance:
                grad_key = _partial_update(
                    grad_key,
                    grad_key_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
                grad_value = _partial_update(
                    grad_value,
                    grad_value_,
                    dim=seq_dim,
                    n_chunks=ROUND_ROBIN_CYCLE,
                    idx=0,
                    add=True,
                )
            else:
                grad_key += grad_key_
                grad_value += grad_value_

        next_grad_kv = torch.cat([grad_key.flatten(), grad_value.flatten()])
        # Send the grad key and grad value to the next rank.
        dkv_rotater.exchange_buffers(next_grad_kv)

        if i <= rank or not _cp_options.enable_load_balance:
            grad_query += grad_query_
        else:
            grad_query = _partial_update(
                grad_query,
                grad_query_,
                dim=seq_dim,
                n_chunks=ROUND_ROBIN_CYCLE,
                idx=1,
                add=True,
            )

    assert grad_key_ is not None
    assert grad_value_ is not None

    grad_query = grad_query.to(query.dtype)
    next_grad_kv = dkv_rotater.next_buffer().to(key.dtype)
    grad_key = next_grad_kv[: grad_key.numel()].reshape(grad_key.shape)
    grad_value = next_grad_kv[grad_key.numel() :].reshape(grad_value.shape)

    return grad_query, grad_key, grad_value, *rest


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
    ) -> torch.Tensor:
        H = q.size(-1)
        if H % 8 != 0:
            pad = 8 - H % 8
            q = torch.nn.functional.pad(q, [0, pad])
            k = torch.nn.functional.pad(k, [0, pad])
            v = torch.nn.functional.pad(v, [0, pad])

        out, logsumexp = _ring_attention_forward(
            q=q,
            k=k,
            v=v,
            causal=causal,
            dropout=dropout,
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
        )

        ctx.save_for_backward(q, k, v, out, logsumexp)
        ctx.dropout = dropout
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        out = out[..., :H]

        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise ValueError
        return super().backward(ctx, *grad_outputs)


def ring_attention_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout: float,
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
) -> tuple[torch.Tensor, ...]:
    return _RingAttention.apply(q, k, v, causal, dropout, softmax_scale, window_size, softcap)
