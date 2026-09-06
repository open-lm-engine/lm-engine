# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch


def _pack_torch(
    x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
) -> torch.Tensor:
    B, S = x.size()[:2]
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

    if padding_side == "left":
        pad_tokens = S - seqlens
        seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
    elif padding_side == "right":
        seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
    else:
        raise ValueError(f"unexpected padding_side ({padding_side})")

    x = x[batch_indices, seq_indices]

    return x


def _unpack_torch(
    x: torch.Tensor, cu_seqlens: torch.Tensor, output_shape: tuple[int], padding_side: str
) -> torch.Tensor:
    B = cu_seqlens.size(0) - 1
    S = output_shape[1]

    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    batch_indices = torch.arange(B, device=x.device).repeat_interleave(seqlens)

    if padding_side == "left":
        pad_tokens = S - seqlens
        seq_indices = torch.cat([torch.arange(sl, S, device=x.device) for sl in pad_tokens])
    elif padding_side == "right":
        seq_indices = torch.cat([torch.arange(sl, device=x.device) for sl in seqlens])
    else:
        raise ValueError(f"unexpected padding_side ({padding_side})")

    padded = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
    padded[batch_indices, seq_indices] = x

    return padded
