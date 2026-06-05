# **************************************************
# Copyright (c) 2026, Mayank Mishra, Han Guo
# **************************************************

import os

import torch
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from fla.ops.cp import FLACPContext, build_cp_context

from lm_engine.hf_models.modeling_utils.mlp_blocks.delta_mlp.shortconv import causal_conv1d_cp


def prepare_data(
    B: int,
    T: int,
    D: int,
    W: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Seed identically on every rank so all ranks see the same inputs
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(B, T, D, generator=g, dtype=dtype, device=device, requires_grad=True)
    weight = torch.randn(D, W, generator=g, dtype=dtype, device=device, requires_grad=True)
    do = torch.randn(B, T, D, generator=g, dtype=dtype, device=device)
    return x, weight, do


def fwd_bwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    do: torch.Tensor,
    cp_context: FLACPContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cp_context is None:
        o = causal_conv1d_fn(
            x=rearrange(x, "b t d -> b d t"),
            weight=weight,
            bias=None,
            activation=None,
        )
        o = rearrange(o, "b d t -> b t d")
    else:
        o = causal_conv1d_cp(
            x=x,
            weight=weight,
            bias=None,
            activation=None,
            cp_context=cp_context,
        )
    dx, dw = torch.autograd.grad(
        outputs=o,
        inputs=(x, weight),
        grad_outputs=do,
    )
    return o, dx, dw


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.group.WORLD
    device = f"cuda:{local_rank}"

    B = 1
    T = 512
    D = 3712
    W = 4

    x, weight, do = prepare_data(
        B=B,
        T=T,
        D=D,
        W=W,
        dtype=torch.float32,
        device=device,
    )

    o0, dx0, dw0 = fwd_bwd(
        x=x,
        weight=weight,
        do=do,
        cp_context=None,
    )

    flat = lambda x: rearrange(x, "b t ... -> 1 (b t) ...")
    tokens_per_rank = (B * T) // world_size
    t0 = rank * tokens_per_rank
    t1 = t0 + tokens_per_rank
    cu_seqlens = torch.tensor(
        [i * T for i in range(B + 1)],
        dtype=torch.long,
        device=device,
    )
    cp_context = build_cp_context(
        cu_seqlens=cu_seqlens,
        group=group,
        conv1d_kernel_size=W,
    )
    o1, dx1, dw1 = fwd_bwd(
        x=flat(x)[:, t0:t1].contiguous(),
        weight=weight,
        do=flat(do)[:, t0:t1].contiguous(),
        cp_context=cp_context,
    )

    # Per-tensor max-abs-diff across all ranks, for diagnostic visibility.
    # Every rank must call this (it does an all_reduce internally); only rank 0 prints.
    def max_abs_diff_across_ranks(a: torch.Tensor, b: torch.Tensor) -> float:
        amax = (a - b).abs().max()
        torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=group)
        return amax.item()

    diffs = {
        name: max_abs_diff_across_ranks(a, b)
        for name, a, b in [
            ("o", o1, flat(o0)[:, t0:t1]),
            ("dx", dx1, flat(dx0)[:, t0:t1]),
        ]
    }
    if rank == 0:
        print("  max abs diff: " + " ".join(f"{n}={v:.2e}" for n, v in diffs.items()))

    # Per-token output + input grad: per-rank CP value matches the reference slice,
    # including the halo boundary that the CP backward routes between neighbours.
    torch.testing.assert_close(o1, flat(o0)[:, t0:t1])
    torch.testing.assert_close(dx1, flat(dx0)[:, t0:t1])

    # Weight grad: every output position is computed on exactly one rank, so summing
    # each rank's shard recovers the full (batch-and-sequence-summed) weight grad.
    torch.distributed.all_reduce(dw1, op=torch.distributed.ReduceOp.SUM, group=group)
    if rank == 0:
        print(f"  max abs diff: dw={(dw1 - dw0).abs().max().item():.2e}")
    torch.testing.assert_close(dw1, dw0)

    if rank == 0:
        print(f"PASS: world={world_size}")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
