# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F


def compute_cu_seqlens_and_max_seqlen_from_attention_mask(attention_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = seqlens.max().item()
    return cu_seqlens, max_seqlen
