# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import pytest


torch = pytest.importorskip("torch")

import torch.nn.functional as F

from lm_engine.accelerator import Accelerator
from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels, is_kernel_allowed
from lm_engine.modeling_utils.mlp_blocks.moe.experts import ColumnParallelExperts, RowParallelExperts
from tests.layers.utils import assert_equal_tensors
from tests.utils import skip_test_if_device_unavailable


_SEED = 42


def _run_experts(
    c_fc: ColumnParallelExperts,
    c_proj: RowParallelExperts,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
) -> torch.Tensor:
    """mirrors MoE._compute_experts (lm_engine/modeling_utils/mlp_blocks/moe/module.py) exactly, without the
    routing/gating machinery, so that the scattermoe and naive-torch code paths of ColumnParallelExperts and
    RowParallelExperts can be compared directly."""

    T = x.size(0)

    sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
    expert_frequency = sorted_expert_idxs.bincount(minlength=num_experts).to(torch.uint32)

    if is_kernel_allowed(Kernel.scattermoe):
        expert_offsets = expert_frequency.cumsum(-1)

        x = c_fc(
            x=x,
            num_experts_per_token=num_experts_per_tok,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
        )
        x = F.gelu(x)
        x = c_proj(
            x=x,
            num_experts_per_token=1,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            expert_offsets=expert_offsets,
            router_weights=router_weights,
        )
    else:
        batch_index = sorted_scattered_idxs // num_experts_per_tok
        batch_gates = router_weights.flatten()[sorted_scattered_idxs]

        x = x[batch_index]
        x = c_fc(x=x, expert_frequency=expert_frequency)
        x = F.gelu(x)
        x = c_proj(x=x, expert_frequency=expert_frequency)

        x = x * batch_gates.unsqueeze(-1)
        zeros = torch.zeros((T, hidden_size), dtype=x.dtype, device=x.device)
        x = zeros.index_add(0, batch_index, x)

    return x


def _generate_args() -> list:
    return list(
        product(
            [torch.float32, torch.float16, torch.bfloat16],  # dtype
            [4, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [7, 128],  # num_tokens
        )
    )


@pytest.mark.parametrize("dtype,num_experts,num_experts_per_tok,num_tokens", _generate_args())
def test_scattermoe_experts_forward_backward(
    dtype: torch.dtype, num_experts: int, num_experts_per_tok: int, num_tokens: int
) -> None:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    Accelerator.set_seed(_SEED)

    hidden_size = 64
    intermediate_size = 128

    with torch.device(device):
        c_fc = ColumnParallelExperts(
            num_experts=num_experts, in_features=hidden_size, out_features=intermediate_size, add_bias=False, std=0.02
        ).to(dtype=dtype)
        c_proj = RowParallelExperts(
            num_experts=num_experts, in_features=intermediate_size, out_features=hidden_size, add_bias=False, std=0.02
        ).to(dtype=dtype)

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_kernel = x.detach().clone().requires_grad_()
    x_torch = x.detach().clone().requires_grad_()

    selected_experts = torch.stack(
        [torch.randperm(num_experts, device=device)[:num_experts_per_tok] for _ in range(num_tokens)]
    )
    router_logits = torch.randn(num_tokens, num_experts_per_tok, device=device, dtype=dtype)
    router_weights = F.softmax(router_logits.float(), dim=-1).type_as(x)

    with enable_kernels([Kernel.scattermoe]):
        y_kernel = _run_experts(
            c_fc, c_proj, x_kernel, router_weights, selected_experts, num_experts, num_experts_per_tok, hidden_size
        )

    y_torch = _run_experts(
        c_fc, c_proj, x_torch, router_weights, selected_experts, num_experts, num_experts_per_tok, hidden_size
    )

    assert_equal_tensors(
        y_kernel,
        y_torch,
        False,
        atol_float32=5.1e-3,
        rtol_float32=0,
        atol_float16=2e-3,
        rtol_float16=0,
        atol_bfloat16=1.6e-2,
        rtol_bfloat16=0,
    )

    y_kernel.sum().backward()
    c_fc_grad_kernel, c_proj_grad_kernel = c_fc.weight.grad, c_proj.weight.grad
    c_fc.weight.grad = None
    c_proj.weight.grad = None

    y_torch.sum().backward()
    c_fc_grad_torch, c_proj_grad_torch = c_fc.weight.grad, c_proj.weight.grad

    assert_equal_tensors(
        x_kernel.grad,
        x_torch.grad,
        False,
        atol_float32=5.9e-3,
        rtol_float32=0,
        atol_float16=2e-3,
        rtol_float16=0,
        atol_bfloat16=1.6e-2,
        rtol_bfloat16=0,
    )

    for grad_kernel, grad_torch in [(c_fc_grad_kernel, c_fc_grad_torch), (c_proj_grad_kernel, c_proj_grad_torch)]:
        assert_equal_tensors(
            grad_kernel,
            grad_torch,
            False,
            atol_float32=3e-2,
            rtol_float32=0,
            atol_float16=2e-3,
            rtol_float16=0,
            atol_bfloat16=7.9e-3,
            rtol_bfloat16=0,
        )
