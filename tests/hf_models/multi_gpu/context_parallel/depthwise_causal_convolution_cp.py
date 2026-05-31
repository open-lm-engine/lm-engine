# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed
from torch.testing import assert_close

from lm_engine.enums import Kernel
from lm_engine.hf_models.modeling_utils import DepthwiseCausalConvolution
from lm_engine.kernels import enable_kernels
from lm_engine.parallel import ProcessGroupManager, prepare_context_parallel_input


parser = argparse.ArgumentParser()
parser.add_argument("--kernel-size", type=int)
parser.add_argument("--use-causal-conv1d", action="store_true")
args = parser.parse_args()

cp_world_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(context_parallel_world_size=cp_world_size)

rank = ProcessGroupManager.get_context_parallel_rank()
device = torch.cuda.current_device()

_HIDDEN_SIZE = 16
_BATCH = 2
_CHUNK_LEN = 32

torch.manual_seed(42)
conv = DepthwiseCausalConvolution(
    hidden_size=_HIDDEN_SIZE,
    kernel_size=args.kernel_size,
    activation_function="silu",
    add_bias=True,
    std=None,
    use_padding_free_transformer=False,
).to(device)
conv.eval()

torch.manual_seed(0)

x_full = torch.randn(_BATCH, cp_world_size * _CHUNK_LEN, _HIDDEN_SIZE, device=device)
x_local = prepare_context_parallel_input((x_full,))[0]

kernels = [Kernel.causal_conv1d] if args.use_causal_conv1d else []
cp_group = ProcessGroupManager.get_context_parallel_group()

# ---- forward ----
with enable_kernels(kernels):
    out_local, _ = conv(x_local.detach(), input_state=None, attention_mask=None, output_state=False)

parts = [torch.zeros_like(out_local) for _ in range(cp_world_size)]
torch.distributed.all_gather(parts, out_local.detach().contiguous(), group=cp_group)
out_cp_full = torch.cat(parts, dim=1)

if rank == 0:
    with enable_kernels(kernels), ProcessGroupManager.set_dummy_context_parallel_world_size(1):
        out_ref, _ = conv(x_full, input_state=None, attention_mask=None, output_state=False)

    assert_close(out_cp_full, out_ref, rtol=1e-5, atol=1e-5)

# ---- backward ----
x_local_bwd = x_local.detach().requires_grad_(True)
with enable_kernels(kernels):
    out_local_bwd, _ = conv(x_local_bwd, input_state=None, attention_mask=None, output_state=False)
    out_local_bwd.sum().backward()

grad_parts = [torch.zeros_like(x_local_bwd) for _ in range(cp_world_size)]
torch.distributed.all_gather(grad_parts, x_local_bwd.grad.contiguous(), group=cp_group)
grad_cp_full = torch.cat(grad_parts, dim=1)

if rank == 0:
    x_full_ref = x_full.detach().requires_grad_(True)
    with enable_kernels(kernels), ProcessGroupManager.set_dummy_context_parallel_world_size(1):
        out_ref_bwd, _ = conv(x_full_ref, input_state=None, attention_mask=None, output_state=False)
        out_ref_bwd.sum().backward()

    assert_close(grad_cp_full, x_full_ref.grad, rtol=1e-5, atol=1e-5)

ProcessGroupManager.destroy_process_groups()
