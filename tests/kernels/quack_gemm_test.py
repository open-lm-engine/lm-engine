# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from typing import NamedTuple

import pytest
import torch
from torch.testing import assert_close as torch_assert_close

from lm_engine.accelerator import Accelerator
from lm_engine.arguments import KernelArgs
from lm_engine.enums import Kernel
from lm_engine.kernels import enable_kernels
from lm_engine.modeling_utils.linear import ParameterizedLinear, linear_func
from lm_engine.modeling_utils.mlp_blocks import MLP
from lm_engine.utils import is_quack_available
from tests.utils import skip_test_if_device_unavailable


SEED = 1234
DEVICE = torch.device("cuda")
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
SHAPE = (8, 16, HIDDEN_SIZE)


_FC1_ACT_CASES = [
    (Kernel.quack_gemm_gated, "swiglu"),
    (Kernel.quack_gemm_gated, "sigmoid_glu"),
    (Kernel.quack_gemm_act, "relu"),
    (Kernel.quack_gemm_act, "gelu_pytorch_tanh"),
]

_EQUIVALENCE_CASES = [
    pytest.param(
        kernel,
        activation_function,
        add_bias,
        dtype,
        id=f"{kernel.value}_{activation_function}_{'bias' if add_bias else 'no_bias'}_{str(dtype).split('.')[-1]}",
    )
    for kernel, activation_function in _FC1_ACT_CASES
    for add_bias in [False, True]
    for dtype in [torch.float16, torch.bfloat16]
]

_LINEAR_EQUIVALENCE_CASES = [
    pytest.param(add_bias, dtype, id=f"{'bias' if add_bias else 'no_bias'}_{str(dtype).split('.')[-1]}")
    for add_bias in [False, True]
    for dtype in [torch.float16, torch.bfloat16]
]

_PLAIN_MLP_EQUIVALENCE_CASES = [
    pytest.param(
        activation_function,
        dtype,
        id=f"{activation_function}_{str(dtype).split('.')[-1]}",
    )
    for activation_function in ["gelu_pytorch_tanh", "swiglu"]
    for dtype in [torch.float16, torch.bfloat16]
]

_COMBINED_MLP_EQUIVALENCE_CASES = [
    pytest.param([Kernel.quack_gemm, Kernel.quack_gemm_act], "gelu_pytorch_tanh", id="quack_gemm_act"),
    pytest.param([Kernel.quack_gemm, Kernel.quack_gemm_gated], "swiglu", id="quack_gemm_gated"),
]


class _LinearResult(NamedTuple):
    y: torch.Tensor
    dx: torch.Tensor
    dw: torch.Tensor
    db: torch.Tensor | None


class _MLPResult(NamedTuple):
    y: torch.Tensor
    dx: torch.Tensor
    dc_fc: torch.Tensor
    dc_proj: torch.Tensor
    db_fc: torch.Tensor | None
    db_proj: torch.Tensor | None


def _make_mlp(activation_function: str, add_bias: bool) -> MLP:
    return MLP(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        activation_function=activation_function,
        dropout=0,
        add_bias=add_bias,
        init_method="normal",
        initializer_range=0.02,
        m_width=1,
        num_layers=1,
        use_depth_scaled_init=False,
    )


def _run_linear_func(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    grad: torch.Tensor,
    kernels: list[Kernel],
) -> _LinearResult:
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    bias = bias.detach().clone().requires_grad_(True) if bias is not None else None

    with enable_kernels(kernels):
        y = linear_func(x, weight, bias)
        y.backward(grad)

    assert x.grad is not None
    assert weight.grad is not None

    return _LinearResult(
        y.detach(),
        x.grad.detach(),
        weight.grad.detach(),
        bias.grad.detach() if bias is not None else None,
    )


def _run_linear_module(
    module: ParameterizedLinear,
    x: torch.Tensor,
    grad: torch.Tensor,
    kernels: list[Kernel],
) -> _LinearResult:
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)

    with enable_kernels(kernels):
        y = module(x)
        y.backward(grad)

    assert x.grad is not None
    assert module.weight.grad is not None

    return _LinearResult(
        y.detach(),
        x.grad.detach(),
        module.weight.grad.detach(),
        module.bias.grad.detach() if module.bias is not None else None,
    )


def _run_mlp(
    module: MLP,
    x: torch.Tensor,
    grad: torch.Tensor,
    kernels: list[Kernel],
) -> _MLPResult:
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)

    with enable_kernels(kernels):
        y = module(x)
        y.backward(grad)

    assert x.grad is not None
    assert module.c_fc.weight.grad is not None
    assert module.c_proj.weight.grad is not None

    return _MLPResult(
        y.detach(),
        x.grad.detach(),
        module.c_fc.weight.grad.detach(),
        module.c_proj.weight.grad.detach(),
        module.c_fc.bias.grad.detach() if module.c_fc.bias is not None else None,
        module.c_proj.bias.grad.detach() if module.c_proj.bias is not None else None,
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
    reference: torch.Tensor | None = None,
) -> None:
    if dtype != torch.bfloat16:
        torch_assert_close(actual, expected, rtol=5e-3, atol=5e-3)
        return

    assert reference is not None

    actual_error = (actual.float() - reference.float()).abs().max().item()
    expected_error = (expected.float() - reference.float()).abs().max().item()

    # Match QuACK's bf16 testing policy: compare the kernel error against a
    # PyTorch bf16 baseline error to the fp32 reference, instead of requiring
    # two low-precision implementations to round identically.
    assert (
        actual_error <= 2 * expected_error + 1e-2
    ), f"quack error {actual_error} exceeds pytorch bf16 baseline error {expected_error}"


@pytest.mark.parametrize("add_bias,dtype", _LINEAR_EQUIVALENCE_CASES)
def test_linear_func_equivalence(add_bias: bool, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=DEVICE, dtype=dtype)
    bias = torch.randn(INTERMEDIATE_SIZE, device=DEVICE, dtype=dtype) if add_bias else None
    grad = torch.randn(*SHAPE[:-1], INTERMEDIATE_SIZE, device=DEVICE, dtype=dtype)

    expected = _run_linear_func(x, weight, bias, grad, [])
    quack = _run_linear_func(x, weight, bias, grad, [Kernel.quack_gemm])
    reference = _run_linear_func(
        x.float(),
        weight.float(),
        bias.float() if bias is not None else None,
        grad.float(),
        [],
    )

    _assert_close(quack.y, expected.y, dtype, reference.y)
    _assert_close(quack.dx, expected.dx, dtype, reference.dx)
    _assert_close(quack.dw, expected.dw, dtype, reference.dw)

    if add_bias:
        assert quack.db is not None
        assert expected.db is not None
        assert reference.db is not None
        _assert_close(quack.db, expected.db, dtype, reference.db)


def test_parameterized_linear_equivalence() -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    dtype = torch.float16
    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    grad = torch.randn(*SHAPE[:-1], INTERMEDIATE_SIZE, device=DEVICE, dtype=dtype)

    module = ParameterizedLinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=True, std=0.02).to(device=DEVICE, dtype=dtype)
    reference_module = ParameterizedLinear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=True, std=0.02).to(
        device=DEVICE, dtype=torch.float32
    )
    reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

    expected = _run_linear_module(module, x, grad, [])
    quack = _run_linear_module(module, x, grad, [Kernel.quack_gemm])
    reference = _run_linear_module(reference_module, x.float(), grad.float(), [])

    _assert_close(quack.y, expected.y, dtype, reference.y)
    _assert_close(quack.dx, expected.dx, dtype, reference.dx)
    _assert_close(quack.dw, expected.dw, dtype, reference.dw)
    assert quack.db is not None
    assert expected.db is not None
    assert reference.db is not None
    _assert_close(quack.db, expected.db, dtype, reference.db)


@pytest.mark.parametrize("activation_function,dtype", _PLAIN_MLP_EQUIVALENCE_CASES)
def test_plain_quack_gemm_mlp_equivalence(activation_function: str, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    grad = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)

    module = _make_mlp(activation_function, add_bias=True).to(device=DEVICE, dtype=dtype)
    reference_module = _make_mlp(activation_function, add_bias=True).to(device=DEVICE, dtype=torch.float32)
    reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

    expected = _run_mlp(module, x, grad, [])
    quack = _run_mlp(module, x, grad, [Kernel.quack_gemm])
    reference = _run_mlp(reference_module, x.float(), grad.float(), [])

    _assert_close(quack.y, expected.y, dtype, reference.y)
    _assert_close(quack.dx, expected.dx, dtype, reference.dx)
    _assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
    _assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)
    assert quack.db_fc is not None
    assert quack.db_proj is not None
    assert expected.db_fc is not None
    assert expected.db_proj is not None
    assert reference.db_fc is not None
    assert reference.db_proj is not None
    _assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
    _assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)


@pytest.mark.parametrize("kernels,activation_function", _COMBINED_MLP_EQUIVALENCE_CASES)
def test_combined_quack_gemm_mlp_equivalence(kernels: list[Kernel], activation_function: str) -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    dtype = torch.float16
    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    grad = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)

    module = _make_mlp(activation_function, add_bias=True).to(device=DEVICE, dtype=dtype)
    reference_module = _make_mlp(activation_function, add_bias=True).to(device=DEVICE, dtype=torch.float32)
    reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

    expected = _run_mlp(module, x, grad, [])
    quack = _run_mlp(module, x, grad, kernels)
    reference = _run_mlp(reference_module, x.float(), grad.float(), [])

    _assert_close(quack.y, expected.y, dtype, reference.y)
    _assert_close(quack.dx, expected.dx, dtype, reference.dx)
    _assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
    _assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)
    assert quack.db_fc is not None
    assert quack.db_proj is not None
    assert expected.db_fc is not None
    assert expected.db_proj is not None
    assert reference.db_fc is not None
    assert reference.db_proj is not None
    _assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
    _assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)


@pytest.mark.parametrize("kernel,activation_function,add_bias,dtype", _EQUIVALENCE_CASES)
def test_equivalence(kernel: Kernel, activation_function: str, add_bias: bool, dtype: torch.dtype) -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    Accelerator.set_seed(SEED)

    x = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)
    grad = torch.randn(*SHAPE, device=DEVICE, dtype=dtype)

    module = _make_mlp(activation_function, add_bias=add_bias).to(device=DEVICE, dtype=dtype)
    reference_module = _make_mlp(activation_function, add_bias=add_bias).to(device=DEVICE, dtype=torch.float32)
    reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

    expected = _run_mlp(module, x, grad, [])
    quack = _run_mlp(module, x, grad, [kernel])
    reference = _run_mlp(reference_module, x.float(), grad.float(), [])

    _assert_close(quack.y, expected.y, dtype, reference.y)
    _assert_close(quack.dx, expected.dx, dtype, reference.dx)
    _assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
    _assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)

    if add_bias:
        assert quack.db_fc is not None
        assert quack.db_proj is not None
        assert expected.db_fc is not None
        assert expected.db_proj is not None
        assert reference.db_fc is not None
        assert reference.db_proj is not None
        _assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
        _assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)


def test_unsupported_gemm_act_activation() -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    module = _make_mlp("gelu", add_bias=False).to(device=DEVICE, dtype=torch.float32)

    with enable_kernels([Kernel.quack_gemm_act]):
        with pytest.raises(ValueError, match="is not supported by quack_gemm_act"):
            module(torch.randn(*SHAPE, device=DEVICE))


def test_unsupported_gemm_gated_activation() -> None:
    skip_test_if_device_unavailable(DEVICE)

    if not is_quack_available():
        pytest.skip("skipping test because quack-kernels is unavailable")

    module = _make_mlp("geglu", add_bias=False).to(device=DEVICE, dtype=torch.float32)

    with enable_kernels([Kernel.quack_gemm_gated]):
        with pytest.raises(ValueError, match="is not supported by quack_gemm_gated"):
            module(torch.randn(*SHAPE, device=DEVICE))


def test_quack_gemm_allows_gemm_act() -> None:
    KernelArgs(kernels=[Kernel.quack_gemm, Kernel.quack_gemm_act])


def test_quack_gemm_allows_gemm_gated() -> None:
    KernelArgs(kernels=[Kernel.quack_gemm, Kernel.quack_gemm_gated])


def test_quack_gemm_act_and_gated_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="quack_gemm_act cannot be enabled with quack_gemm_gated"):
        KernelArgs(kernels=[Kernel.quack_gemm_act, Kernel.quack_gemm_gated])
