# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum
from typing import Callable

import torch

from .....parallel import ProcessGroupManager
from ...rotaters import AllToAllRotater
from .merge import _Merger, _partial_update


class _CausalBehavior(Enum):
    SKIP = None
    NOT_IS_CAUSAL = False
    IS_CAUSAL = True

    @staticmethod
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

        return _CausalBehavior.SKIP


def _ring_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
    forward_function: Callable,
) -> tuple[torch.Tensor, ...]:
    BLOCK_SIZE_S = q.size(1)

    if causal and k.size(1) != BLOCK_SIZE_S:
        raise NotImplementedError("causal requires the same query and context sequence lengths")

    if not causal and ProcessGroupManager.get_context_parallel_load_balancing_method() is not None:
        raise RuntimeError("Load balancing requires `causal=True`.")

    rank = ProcessGroupManager.get_context_parallel_rank()
    world_size = ProcessGroupManager.get_context_parallel_world_size()

    use_sliding_window = window_size != (-1, -1)

    if use_sliding_window:
        assert window_size[0] == window_size[1]
        assert causal
        assert ProcessGroupManager.get_context_parallel_load_balancing_method() is None

        num_loops = min(world_size, (window_size[0] + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S + 1)
    else:
        num_loops = world_size

    # Without making key and value contiguous(), the loss curve is bad.
    # TODO(fegin): figure out why this is a requirement since SDPA does not have
    # this requirement.
    k = k.contiguous()
    v = v.contiguous()

    # Save original shapes so buffer slicing is correct even after chunking.
    k_size = k.size()
    v_size = v.size()

    rotater = AllToAllRotater()
    sdpa_merger = _Merger(1)

    for i in range(num_loops):
        is_reversed_computation = False

        if use_sliding_window:
            local_sliding_window = window_size[0] - i * BLOCK_SIZE_S

        if i > 0:
            is_reversed_computation = use_sliding_window and i == num_loops - 1 and local_sliding_window <= 0
            _k_size = k_size
            _v_size = v_size

            if is_reversed_computation:
                reversed_seq_len = BLOCK_SIZE_S + local_sliding_window

                _k_size = list(k_size)
                _v_size = list(v_size)

                _k_size[1] = reversed_seq_len
                _v_size[1] = reversed_seq_len

            k, v = rotater.next_buffer().chunk(2)
            k = k.reshape(_k_size)
            v = v.reshape(_v_size)

        if i < num_loops - 1:
            k_send = k
            v_send = v

            if use_sliding_window and i == num_loops - 2 and local_sliding_window <= BLOCK_SIZE_S:
                assert local_sliding_window > 0
                # send sliced and reversed
                k_send = k_send[:, -local_sliding_window:].flip([1])
                v_send = v_send[:, -local_sliding_window:].flip([1])

            rotater.exchange_buffers(torch.cat([k_send.flatten(), v_send.flatten()]), with_grad=False)

        is_causal_behavior = _CausalBehavior._is_causal_behavior(rank=rank, world_size=world_size, i=i, causal=causal)

        if is_reversed_computation and is_causal_behavior != _CausalBehavior.SKIP:
            is_causal_behavior = _CausalBehavior.IS_CAUSAL

        if is_causal_behavior == _CausalBehavior.SKIP:
            continue

        if is_reversed_computation:
            # The last partial window uses a Q prefix and K/V suffix. Reversing
            # both lets a causal mask express the upper-diagonal boundary.
            local_q = q[:, :reversed_seq_len].flip([1])
            local_k = k
            local_v = v
            partial = False
        elif i == 0 or ProcessGroupManager.get_context_parallel_load_balancing_method() is None or not causal:
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

        window_size_left = -1 if is_reversed_computation or not use_sliding_window else local_sliding_window
        window_size_right = (
            window_size[1]
            if (use_sliding_window and not is_reversed_computation and is_causal_behavior == _CausalBehavior.IS_CAUSAL)
            else -1
        )

        # TODO use SM margin for better overlapping of communication
        x, lse, _, _ = forward_function(
            q=local_q,
            k=local_k,
            v=local_v,
            softmax_scale=softmax_scale,
            causal=is_causal_behavior == _CausalBehavior.IS_CAUSAL,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            softcap=softcap,
        )

        if is_reversed_computation:
            x = x.flip([1])
            lse = lse.flip([-1])

            if reversed_seq_len != BLOCK_SIZE_S:
                x_full = torch.zeros_like(q)
                lse_full = lse.new_full((lse.size(0), lse.size(1), BLOCK_SIZE_S), float("-inf"))

                x_full[:, :reversed_seq_len] = x
                lse_full[:, :, :reversed_seq_len] = lse

                x = x_full
                lse = lse_full

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
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
    backward_function: Callable,
) -> tuple[torch.Tensor, ...]:
    BLOCK_SIZE_S = q.size(1)
    rank = ProcessGroupManager.get_context_parallel_rank()
    world_size = ProcessGroupManager.get_context_parallel_world_size()

    use_sliding_window = window_size != (-1, -1)
    if use_sliding_window:
        num_loops = min(world_size, (window_size[0] + BLOCK_SIZE_S - 1) // BLOCK_SIZE_S + 1)
    else:
        num_loops = world_size

    next_kv = None
    next_grad_kv = None

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    k = k.contiguous()
    v = v.contiguous()

    k_numel = k.numel()
    k_size = k.size()
    v_size = v.size()

    kv_rotater = AllToAllRotater()
    dkv_rotater = None if use_sliding_window else AllToAllRotater()

    for i in range(num_loops):
        if i > 0:
            # Wait for the kv from the (cp_rank - 1) rank.
            buffer = kv_rotater.next_buffer()
            k = buffer[:k_numel].reshape(k_size)
            v = buffer[k_numel:].reshape(v_size)

        if i != num_loops - 1:
            # Send the kv to the next rank.
            next_kv = torch.cat([k.flatten(), v.flatten()])
            kv_rotater.exchange_buffers(next_kv, with_grad=False)

        is_causal_behavior = _CausalBehavior._is_causal_behavior(rank=rank, world_size=world_size, i=i, causal=causal)
        local_sliding_window = window_size[0] - i * BLOCK_SIZE_S if use_sliding_window else -1
        is_reversed_computation = use_sliding_window and i > 0 and i == num_loops - 1 and local_sliding_window <= 0

        if is_reversed_computation and is_causal_behavior != _CausalBehavior.SKIP:
            is_causal_behavior = _CausalBehavior.IS_CAUSAL

        # Skip blocks outside the sliding window (i >= num_loops) in addition to the
        # standard causal SKIP. The dkv rotation still runs for all world_size steps so
        # that gradients route correctly back to their source ranks.
        should_skip = is_causal_behavior == _CausalBehavior.SKIP or (use_sliding_window and i >= num_loops)

        if should_skip:
            dq_ = None
            dk_ = None
            dv_ = None
        else:
            if is_reversed_computation:
                reversed_seq_len = BLOCK_SIZE_S + local_sliding_window

                # Mirror the reversed forward subproblem, then scatter its
                # gradients back into the original Q prefix and K/V suffix.
                local_q = q[:, :reversed_seq_len].flip([1])
                local_k = k[:, -reversed_seq_len:].flip([1])
                local_v = v[:, -reversed_seq_len:].flip([1])
                local_x = x[:, :reversed_seq_len].flip([1])
                local_dx = dx[:, :reversed_seq_len].flip([1])
                local_lse = lse[:, :, :reversed_seq_len].flip([-1]).contiguous()
            elif i == 0 or ProcessGroupManager.get_context_parallel_load_balancing_method() is None or not causal:
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
                # lse layout is [B, H, S]; chunk along the seq dim.
                local_lse = lse.chunk(2, dim=2)[1].contiguous()

            dq_ = torch.empty_like(local_q)
            dk_ = torch.empty_like(local_k)
            dv_ = torch.empty_like(local_v)

            # See https://github.com/pytorch/pytorch/blob/release/2.4/aten/src/ATen/native/native_functions.yaml#L14695
            # for the SDPA kernel definitions.
            backward_function(
                dout=local_dx,
                q=local_q,
                k=local_k,
                v=local_v,
                out=local_x,
                softmax_lse=local_lse,
                dq=dq_,
                dk=dk_,
                dv=dv_,
                softmax_scale=softmax_scale,
                is_causal=is_causal_behavior == _CausalBehavior.IS_CAUSAL,
                window_size_left=-1 if is_reversed_computation else local_sliding_window,
                window_size_right=(
                    window_size[1]
                    if (
                        use_sliding_window
                        and not is_reversed_computation
                        and is_causal_behavior == _CausalBehavior.IS_CAUSAL
                    )
                    else -1
                ),
                softcap=softcap,
            )

            if is_reversed_computation:
                dq_full = torch.zeros_like(q)
                dk_full = torch.zeros_like(k)
                dv_full = torch.zeros_like(v)

                dq_full[:, :reversed_seq_len] = dq_.flip([1])
                dk_full[:, -reversed_seq_len:] = dk_.flip([1])
                dv_full[:, -reversed_seq_len:] = dv_.flip([1])

                dq_ = dq_full
                dk_ = dk_full
                dv_ = dv_full

        if use_sliding_window:
            if dq_ is not None:
                dq += dq_

            if i == 0:
                if dk_ is not None:
                    dk += dk_.to(dtype=dk.dtype)

                if dv_ is not None:
                    dv += dv_.to(dtype=dv.dtype)
            else:
                # Only ranks within num_loops can contribute in sliding-window mode.
                # Send each contribution directly back to its K/V owner instead of
                # rotating empty accumulators around the rest of the ring.
                if dk_ is None:
                    dk_send = torch.zeros_like(dk)
                    dv_send = torch.zeros_like(dv)
                else:
                    dk_send = dk_.to(dtype=dk.dtype)
                    dv_send = dv_.to(dtype=dv.dtype)

                grad_kv_rotater = AllToAllRotater()
                grad_kv_rotater.exchange_buffers(
                    torch.cat([dk_send.flatten(), dv_send.flatten()]), with_grad=False, shift=-i
                )
                grad_kv = grad_kv_rotater.next_buffer()
                dk += grad_kv[: dk.numel()].reshape(dk.size())
                dv += grad_kv[dk.numel() :].reshape(dv.size())

            continue

        ROUND_ROBIN_CYCLE = 2
        if i == 0:
            if dk_ is not None:
                dk += dk_

            if dv_ is not None:
                dv += dv_
        else:
            # Wait for the kv gradient from (cp_rank - 1) rank.
            assert dkv_rotater is not None
            next_grad_kv = dkv_rotater.next_buffer()
            dk = next_grad_kv[: dk.numel()].reshape(dk.size())
            dv = next_grad_kv[dk.numel() :].reshape(dv.size())

            if i <= rank and ProcessGroupManager.get_context_parallel_load_balancing_method() is not None:
                dk = _partial_update(dk, dk_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=0, add=True)
                dv = _partial_update(dv, dv_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=0, add=True)
            else:
                if dk_ is not None:
                    dk += dk_

                if dv_ is not None:
                    dv += dv_

        next_grad_kv = torch.cat([dk.flatten(), dv.flatten()])
        # Send the grad key and grad value to the next rank.
        assert dkv_rotater is not None
        dkv_rotater.exchange_buffers(next_grad_kv, with_grad=False)

        if i <= rank or ProcessGroupManager.get_context_parallel_load_balancing_method() is None:
            if dq_ is not None:
                dq += dq_
        else:
            dq = _partial_update(dq, dq_, dim=1, n_chunks=ROUND_ROBIN_CYCLE, idx=1, add=True)

    dq = dq.type_as(q)
    if use_sliding_window:
        dk = dk.type_as(k)
        dv = dv.type_as(v)
    else:
        assert dkv_rotater is not None
        next_grad_kv = dkv_rotater.next_buffer().type_as(k)
        dk = next_grad_kv[: dk.numel()].reshape(dk.size())
        dv = next_grad_kv[dk.numel() :].reshape(dv.size())

    return dq, dk, dv


class _RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
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
            softmax_scale=softmax_scale,
            window_size=window_size,
            softcap=softcap,
            forward_function=forward_function,
        )

        ctx.save_for_backward(q, k, v, x, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.backward_function = backward_function
        ctx.head_dim = H

        x = x[..., :H]

        return x

    @staticmethod
    def backward(ctx, dx: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, x, lse = ctx.saved_tensors

        H = ctx.head_dim
        H_padded = q.size(-1)
        if H_padded != H:
            dx = torch.nn.functional.pad(dx, [0, H_padded - H])

        dq, dk, dv = _ring_attention_backward(
            q=q,
            k=k,
            v=v,
            x=x,
            dx=dx,
            lse=lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            backward_function=ctx.backward_function,
        )

        if H_padded != H:
            dq = dq[..., :H]
            dk = dk[..., :H]
            dv = dv[..., :H]

        return dq, dk, dv, *[None] * 7


def ring_attention_function(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    softmax_scale: float | None,
    window_size: tuple[int, int],
    softcap: float,
    forward_function: Callable,
    backward_function: Callable,
) -> tuple[torch.Tensor, ...]:
    return _RingAttention.apply(
        q, k, v, causal, softmax_scale, window_size, softcap, forward_function, backward_function
    )
