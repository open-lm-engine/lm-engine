# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from ....parallel import ProcessGroupManager


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

    return input_tensor


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def _all_gather_context_parallel_with_grad(input_tensor: torch.Tensor) -> torch.Tensor:
    cp_mesh = ProcessGroupManager.get_context_parallel_mesh()
    dtensor = DTensor.from_local(input_tensor.contiguous(), device_mesh=cp_mesh, placements=[Shard(0)])
    dtensor = dtensor.redistribute(placements=[Replicate()])

    return dtensor.to_local(grad_placements=[Partial()])


class _SerialPrefixScan(torch.autograd.Function):
    """Serial prefix scan over CP ranks with manual backward.

    Forward:  s[r] = exp_A[r] * s[r-1] + final[r],  s[-1] = 0
    Backward: chain-rule through the linear recurrence without re-entering autograd.
    """

    @staticmethod
    def forward(ctx, all_exp_A: torch.Tensor, all_final: torch.Tensor, cp_rank: int) -> torch.Tensor:
        # all_exp_A : [cp_world_size, batch, num_heads]
        # all_final : [cp_world_size, batch, num_heads, head_dim, state_size]
        prev_states = []
        s = torch.zeros_like(all_final[0])
        for r in range(cp_rank):
            prev_states.append(s)
            s = all_exp_A[r][:, :, None, None] * s + all_final[r]
        ctx.save_for_backward(all_exp_A, *prev_states)
        ctx.cp_rank = cp_rank
        ctx.all_final_shape = all_final.shape
        return s

    @staticmethod
    def backward(ctx, grad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        all_exp_A = ctx.saved_tensors[0]
        prev_states = ctx.saved_tensors[1:]
        cp_rank = ctx.cp_rank

        grad_all_exp_A = torch.zeros_like(all_exp_A)
        grad_all_final = torch.zeros(ctx.all_final_shape, dtype=grad_s.dtype, device=grad_s.device)

        for r in range(cp_rank - 1, -1, -1):
            grad_all_final[r] = grad_s
            grad_all_exp_A[r] = (grad_s * prev_states[r]).sum(dim=(-2, -1))
            grad_s = grad_s * all_exp_A[r][:, :, None, None]

        return grad_all_exp_A, grad_all_final, None


def _serial_prefix_scan(all_exp_A: torch.Tensor, all_final: torch.Tensor, cp_rank: int) -> torch.Tensor:
    return _SerialPrefixScan.apply(all_exp_A, all_final, cp_rank)
