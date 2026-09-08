# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from itertools import product

import pytest
import torch
import torch.nn.functional as F

from lm_engine.accelerator import Accelerator
from lm_engine.training.enums import Kernel
from lm_engine.training.kernels import enable_kernels, is_kernel_allowed
from lm_engine.training.modeling_utils import AttentionMaskInfo
from lm_engine.training.modeling_utils.activations import get_activation_function
from lm_engine.training.modeling_utils.mlp_blocks.moe.experts import ColumnParallelExperts, RowParallelExperts
from tests.utils import (
    assert_equal_tensors,
    from_config,
    get_dummy_inputs,
    get_moe_test_config,
    skip_test_if_device_unavailable,
)


_SEED = 42
_NUM_TOKENS = 7


def _run_experts(
    c_fc: ColumnParallelExperts,
    c_proj: RowParallelExperts,
    act: torch.nn.Module,
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
        x = act(x)
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
        x = act(x)
        x = c_proj(x=x, expert_frequency=expert_frequency)

        x = x * batch_gates.unsqueeze(-1)
        zeros = torch.zeros((T, hidden_size), dtype=x.dtype, device=x.device)
        x = zeros.index_add(0, batch_index, x)

    return x


def _generate_args() -> list:
    # mirrors xma's own tests/layers/moe_test.py::_generate_args (minus kernel_backend/is_compiling, which
    # don't apply here - lm_engine dispatches scattermoe via Kernel.scattermoe, not a KernelBackend argument)
    args = list(
        product(
            [torch.float32, torch.float16],  # dtype
            [2, 4, 6, 8],  # num_experts
            [2, 4],  # num_experts_per_tok
            [2048],  # hidden_size
            [8192],  # intermediate_size
            [True, False],  # is_glu
        )
    )

    args += list(
        product(
            [torch.float32, torch.float16],  # dtype
            [128],  # num_experts
            [8],  # num_experts_per_tok
            [576],  # hidden_size
            [256],  # intermediate_size
            [True, False],  # is_glu
        )
    )

    return args


@pytest.mark.parametrize(
    "dtype,num_experts,num_experts_per_tok,hidden_size,intermediate_size,is_glu", _generate_args()
)
def test_scattermoe_experts_forward_backward(
    dtype: torch.dtype,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    intermediate_size: int,
    is_glu: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    # scatter_kernel.py/group_backward_kernel.py's tl.dot calls read ALLOW_TF32 from
    # torch.backends.cuda.matmul.allow_tf32 at call time, same as the naive torch reference path - disabling
    # it here isolates whether the observed float32 mismatches come from TF32 rounding rather than a real
    # correctness bug
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)

    if num_experts_per_tok > num_experts:
        pytest.skip(
            f"skipping test since number of experts per token ({num_experts_per_tok}) is more than number of "
            f"experts ({num_experts})"
        )

    Accelerator.set_seed(_SEED)

    activation_function = "swiglu" if is_glu else "gelu"
    act = get_activation_function(activation_function)

    with torch.device(device):
        c_fc = ColumnParallelExperts(
            num_experts=num_experts,
            in_features=hidden_size,
            out_features=2 * intermediate_size if is_glu else intermediate_size,
            add_bias=False,
            std=0.02,
        ).to(dtype=dtype)
        c_proj = RowParallelExperts(
            num_experts=num_experts, in_features=intermediate_size, out_features=hidden_size, add_bias=False, std=0.02
        ).to(dtype=dtype)

    x = torch.randn(_NUM_TOKENS, hidden_size, device=device, dtype=dtype, requires_grad=True)
    x_kernel = x.detach().clone().requires_grad_()
    x_torch = x.detach().clone().requires_grad_()

    selected_experts = torch.stack(
        [torch.randperm(num_experts, device=device)[:num_experts_per_tok] for _ in range(_NUM_TOKENS)]
    )
    router_logits = torch.randn(_NUM_TOKENS, num_experts_per_tok, device=device, dtype=dtype)
    router_weights = F.softmax(router_logits.float(), dim=-1).type_as(x)

    with enable_kernels([Kernel.scattermoe]):
        y_kernel = _run_experts(
            c_fc,
            c_proj,
            act,
            x_kernel,
            router_weights,
            selected_experts,
            num_experts,
            num_experts_per_tok,
            hidden_size,
        )

    y_torch = _run_experts(
        c_fc, c_proj, act, x_torch, router_weights, selected_experts, num_experts, num_experts_per_tok, hidden_size
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
        atol_float16=4e-3,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_scattermoe(dtype: torch.dtype) -> None:
    device = torch.device("cuda")
    skip_test_if_device_unavailable(device)

    Accelerator.set_seed(1234)

    input_ids, attention_mask, _ = get_dummy_inputs(device)

    config = get_moe_test_config("rope", num_layers=1, add_bias=False)

    model = from_config(config, dtype=dtype).to(device)
    model.eval()

    naive_output = model(input_ids=input_ids, attention_mask_info=AttentionMaskInfo(attention_mask=attention_mask))

    with enable_kernels([Kernel.scattermoe]):
        scatter_output = model(
            input_ids=input_ids, attention_mask_info=AttentionMaskInfo(attention_mask=attention_mask)
        )

    assert_equal_tensors(
        naive_output.logits,
        scatter_output.logits,
        False,
        rtol_float32=1e-3,
        atol_float32=2e-4,
        rtol_float16=0,
        atol_float16=2.5e-4,
        rtol_bfloat16=0,
        atol_bfloat16=2e-3,
    )
