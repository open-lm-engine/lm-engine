# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed
from torch.testing import assert_close

from lm_engine.enums import ContextParallelLoadBalancerMethod, Kernel
from lm_engine.hf_models.modeling_utils import flash_attention
from lm_engine.kernels import enable_kernels
from lm_engine.parallel import ProcessGroupManager, prepare_context_parallel_input
from lm_engine.parallel.context_parallel import _HeadTailLoadBalancer, _NoLoadBalancer


parser = argparse.ArgumentParser()
parser.add_argument("--attention-implementation", type=str)
parser.add_argument("--load-balancing-method", type=ContextParallelLoadBalancerMethod, default=None)
parser.add_argument("--sliding-window", type=int, default=None)
args = parser.parse_args()

cp_world_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(
    context_parallel_world_size=cp_world_size,
    context_parallel_load_balancing_method=args.load_balancing_method,
)

rank = ProcessGroupManager.get_context_parallel_rank()
device = torch.cuda.current_device()

_BATCH = 2
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_HEAD_DIM = 64
# Fixed chunk per rank so BLOCK_SIZE_S = _CHUNK_LEN regardless of world size.
# This ensures window_size_left = sliding_window - i*BLOCK_SIZE_S >= 0 for all loop
# iterations, since our smallest tested sliding_window (128) equals _CHUNK_LEN.
# Divisible by 2 so headtail load balancing (which needs SEQ_LEN % (2*cp) == 0) works.
_CHUNK_LEN = 128
_SEQ_LEN = _CHUNK_LEN * cp_world_size

kernels = []
if args.attention_implementation == "flash_attention_2":
    kernels = [Kernel.flash_attention_2]
elif args.attention_implementation == "flash_attention_3":
    kernels = [Kernel.flash_attention_3]
elif args.attention_implementation == "flash_attention_4":
    kernels = [Kernel.flash_attention_4]

torch.manual_seed(42)
q_full = torch.randn(_BATCH, _SEQ_LEN, _NUM_Q_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
k_full = torch.randn(_BATCH, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)
v_full = torch.randn(_BATCH, _SEQ_LEN, _NUM_KV_HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16)

# clone before prepare_context_parallel_input modifies in-place so rank-0 keeps the original order
q_ref = q_full.clone()
k_ref = k_full.clone()
v_ref = v_full.clone()

q_local, k_local, v_local = prepare_context_parallel_input(inputs=(q_full, k_full, v_full))

cp_group = ProcessGroupManager.get_context_parallel_group()

_LB_CLASSES = {None: _NoLoadBalancer, ContextParallelLoadBalancerMethod.headtail: _HeadTailLoadBalancer}
lb = _LB_CLASSES[args.load_balancing_method](_SEQ_LEN, cp_world_size, device)
restore_indices = lb._generate_indices(restore=True).squeeze(0).long()


def _gather_and_restore(local: torch.Tensor) -> torch.Tensor:
    parts = [torch.zeros_like(local) for _ in range(cp_world_size)]
    torch.distributed.all_gather(parts, local.detach().contiguous(), group=cp_group)
    gathered = torch.cat(parts, dim=1)
    return gathered[:, restore_indices, ...]


# ---- forward ----
with enable_kernels(kernels):
    out_local = flash_attention(
        q=q_local.detach(),
        k=k_local.detach(),
        v=v_local.detach(),
        attention_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        use_padding_free_transformer=False,
        causal=True,
        sliding_window=args.sliding_window,
    )

out_cp_full = _gather_and_restore(out_local)

if rank == 0:
    with enable_kernels(kernels), ProcessGroupManager.set_dummy_context_parallel_world_size(1):
        out_ref_fwd = flash_attention(
            q=q_ref,
            k=k_ref,
            v=v_ref,
            attention_mask=None,
            cu_seqlens=None,
            max_seqlen=None,
            use_padding_free_transformer=False,
            causal=True,
            sliding_window=args.sliding_window,
        )

    assert_close(out_cp_full, out_ref_fwd, atol=2e-3, rtol=1e-2)

# ---- backward ----
q_bwd = q_local.detach().requires_grad_(True)
k_bwd = k_local.detach().requires_grad_(True)
v_bwd = v_local.detach().requires_grad_(True)

with enable_kernels(kernels):
    out_local_bwd = flash_attention(
        q=q_bwd,
        k=k_bwd,
        v=v_bwd,
        attention_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        use_padding_free_transformer=False,
        causal=True,
        sliding_window=args.sliding_window,
    )
    out_local_bwd.sum().backward()

dq_cp_full = _gather_and_restore(q_bwd.grad)
dk_cp_full = _gather_and_restore(k_bwd.grad)
dv_cp_full = _gather_and_restore(v_bwd.grad)

if rank == 0:
    q_bwd_ref = q_ref.detach().requires_grad_(True)
    k_bwd_ref = k_ref.detach().requires_grad_(True)
    v_bwd_ref = v_ref.detach().requires_grad_(True)

    with enable_kernels(kernels), ProcessGroupManager.set_dummy_context_parallel_world_size(1):
        out_ref_bwd = flash_attention(
            q=q_bwd_ref,
            k=k_bwd_ref,
            v=v_bwd_ref,
            attention_mask=None,
            cu_seqlens=None,
            max_seqlen=None,
            use_padding_free_transformer=False,
            causal=True,
            sliding_window=args.sliding_window,
        )
        out_ref_bwd.sum().backward()

    assert_close(dq_cp_full, q_bwd_ref.grad, atol=1e-2, rtol=1e-2)
    assert_close(dk_cp_full, k_bwd_ref.grad, atol=1e-2, rtol=1e-2)
    assert_close(dv_cp_full, v_bwd_ref.grad, atol=1e-2, rtol=1e-2)

ProcessGroupManager.destroy_process_groups()
