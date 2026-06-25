# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import argparse
import os

import torch
import torch.distributed
from torch.testing import assert_close

from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.sequence_mixer_blocks.mamba2 import Mamba2, Mamba2Args
from lm_engine.parallel import ProcessGroupManager, prepare_context_parallel_input


parser = argparse.ArgumentParser()
parser.add_argument("--use-mamba2-ssm", action="store_true")
args = parser.parse_args()

cp_world_size = int(os.getenv("WORLD_SIZE"))
ProcessGroupManager(context_parallel_world_size=cp_world_size)

rank = ProcessGroupManager.get_context_parallel_rank()
device = torch.cuda.current_device()

_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_NUM_HEADS = 8
_STATE_SIZE = 16
_N_GROUPS = 1
_CHUNK_LEN = 32  # sequence length per CP rank; must be a multiple of chunk_size
_BATCH = 2

config = Mamba2Args(
    state_size=_STATE_SIZE,
    intermediate_size=_INTERMEDIATE_SIZE,
    num_heads=_NUM_HEADS,
    conv_kernel_size=4,
    activation_function="silu",
    num_groups=_N_GROUPS,
    chunk_size=16,
    normalization_function="rmsnorm",
)

torch.manual_seed(42)
mamba2 = Mamba2(
    hidden_size=_HIDDEN_SIZE,
    config=config,
    layer_norm_epsilon=1e-5,
    initializer_range=0.02,
    m_width=1.0,
    init_method="normal",
    num_layers=1,
    layer_idx=0,
    use_depth_scaled_init=False,
).to(device)
mamba2.eval()

torch.manual_seed(0)
x_full = torch.randn(_BATCH, cp_world_size * _CHUNK_LEN, _HIDDEN_SIZE, device=device)
x_local = prepare_context_parallel_input((x_full,))[0]

kernels = [Kernel.mamba2_ssm] if args.use_mamba2_ssm else []
cp_group = ProcessGroupManager.get_context_parallel_group()

# ---- forward ----
with enable_kernels(kernels):
    out_local = mamba2(x_local.detach())

parts = [torch.zeros_like(out_local) for _ in range(cp_world_size)]
torch.distributed.all_gather(parts, out_local.detach().contiguous(), group=cp_group)
out_cp_full = torch.cat(parts, dim=1)

if rank == 0:
    with enable_kernels(kernels), ProcessGroupManager.set_dummy_context_parallel_world_size(1):
        out_ref = mamba2(x_full.detach())

    assert_close(out_cp_full, out_ref)

# ---- backward (smoke test: verify no crash and grads are populated) ----
x_local_bwd = x_local.detach().requires_grad_(True)
with enable_kernels(kernels):
    out_local_bwd = mamba2(x_local_bwd)
    out_local_bwd.sum().backward()

assert x_local_bwd.grad is not None

ProcessGroupManager.destroy_process_groups()
