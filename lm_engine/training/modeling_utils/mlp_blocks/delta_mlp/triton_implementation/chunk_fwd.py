# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import triton
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
from fla.ops.utils import prepare_chunk_indices

from .wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd_intra(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    GDN intra-chunk forward: fused kkt + solve_tril + recompute_w_u.

    Equivalent to:
        A = chunk_scaled_dot_kkt_fwd(k, g, beta, ...)       # kernel 1
        A = solve_tril(A, ...)                              # kernel 2
        w, _ = recompute_w_u_fwd(k, v, beta, A, g, ...)     # kernel 3

    Fuses kernels 1+2 into a single kernel, reducing from 3 to 2 kernel launches
    and eliminating the HBM round-trip for the intermediate A matrix.

    Args:
        k (torch.Tensor):
            The key tensor of shape `[B, T, H, K]`.
        v (torch.Tensor):
            The value tensor of shape `[B, T, HV, V]`.
        g (torch.Tensor):
            The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
        beta (torch.Tensor):
            The beta tensor of shape `[B, T, H]`.
        cu_seqlens (torch.LongTensor):
            The cumulative sequence lengths. Default: `None`.
        chunk_size (int):
            The chunk size. Default: 64.
        chunk_indices (torch.LongTensor):
            Precomputed chunk indices. Default: `None`.

    Returns:
        w (torch.Tensor): shape `[B, T, H, K]`
        A (torch.Tensor): shape `[B, T, H, BT]`, the solved (I+A)^{-1} matrix
    """
    B, T, H, K = k.shape
    assert beta is not None and beta.shape[2] == H
    assert g is None or g.shape[2] == H
    BT = chunk_size
    BC = 16

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Step 1: fused kkt + solve_tril
    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=H,
        K=K,
        BT=BT,
        BC=BC,
        USE_EXP2=use_exp2,
    )

    # Step 2: recompute_w_u
    w, _ = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=use_exp2,
        compute_u=False,
    )
    return w, A
