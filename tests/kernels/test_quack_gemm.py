# **************************************************
# Copyright (c) 2026, Mayank Mishra, Zhonglin Han
# **************************************************

from functools import partial
from typing import NamedTuple

import pytest
import torch
from absl.testing import absltest, parameterized
from torch.testing import assert_close as torch_assert_close

from lm_engine.accelerator import Accelerator
from lm_engine.arguments import KernelArgs
from lm_engine.enums import Kernel
from lm_engine.hf_models.modeling_utils.linear import ParameterizedLinear, linear_func
from lm_engine.hf_models.modeling_utils.mlp_blocks import MLP
from lm_engine.kernels import enable_kernels
from lm_engine.utils import is_quack_available
from tests.utils import skip_test_if_device_unavailable


SEED = 1234
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
SHAPE = (8, 16, HIDDEN_SIZE)


_FC1_ACT_CASES = [
    (Kernel.quack_gemm_gated, "swiglu", True),
    (Kernel.quack_gemm_gated, "sigmoid_glu", True),
    (Kernel.quack_gemm_act, "relu", False),
    (Kernel.quack_gemm_act, "gelu_pytorch_tanh", False),
]

_EQUIVALENCE_CASES = [
    {
        "testcase_name": (
            f"_{kernel.value}_{activation_function}_{'bias' if add_bias else 'no_bias'}_"
            f"{str(dtype).split('.')[-1]}"
        ),
        "kernel": kernel,
        "activation_function": activation_function,
        "use_interleaved_weights": use_interleaved_weights,
        "add_bias": add_bias,
        "dtype": dtype,
    }
    for kernel, activation_function, use_interleaved_weights in _FC1_ACT_CASES
    for add_bias in [False, True]
    for dtype in [torch.float16, torch.bfloat16]
]

_LINEAR_EQUIVALENCE_CASES = [
    {
        "testcase_name": f"_{'bias' if add_bias else 'no_bias'}_{str(dtype).split('.')[-1]}",
        "add_bias": add_bias,
        "dtype": dtype,
    }
    for add_bias in [False, True]
    for dtype in [torch.float16, torch.bfloat16]
]

_PLAIN_MLP_EQUIVALENCE_CASES = [
    {
        "testcase_name": f"_{activation_function}_{str(dtype).split('.')[-1]}",
        "activation_function": activation_function,
        "use_interleaved_weights": use_interleaved_weights,
        "dtype": dtype,
    }
    for activation_function, use_interleaved_weights in [("gelu_pytorch_tanh", False), ("swiglu", True)]
    for dtype in [torch.float16, torch.bfloat16]
]

_COMBINED_MLP_EQUIVALENCE_CASES = [
    {
        "testcase_name": "_quack_gemm_act",
        "kernels": [Kernel.quack_gemm, Kernel.quack_gemm_act],
        "activation_function": "gelu_pytorch_tanh",
        "use_interleaved_weights": False,
    },
    {
        "testcase_name": "_quack_gemm_gated",
        "kernels": [Kernel.quack_gemm, Kernel.quack_gemm_gated],
        "activation_function": "swiglu",
        "use_interleaved_weights": True,
    },
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


class QuackGemmTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.device("cuda")
        self.mlp_fn = partial(
            MLP,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dropout=0,
            init_method="normal",
            initializer_range=0.02,
            m_width=1,
            num_layers=1,
            use_depth_scaled_init=False,
        )

    def _run_linear_func(
        self,
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
        self,
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
        self,
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

    def assert_close(
        self,
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
        self.assertLessEqual(
            actual_error,
            2 * expected_error + 1e-2,
            f"quack error {actual_error} exceeds pytorch bf16 baseline error {expected_error}",
        )

    @parameterized.named_parameters(*_LINEAR_EQUIVALENCE_CASES)
    def test_linear_func_equivalence(self, add_bias: bool, dtype: torch.dtype) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        Accelerator.set_seed(SEED)

        x = torch.randn(*SHAPE, device=self.device, dtype=dtype)
        weight = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=self.device, dtype=dtype)
        bias = torch.randn(INTERMEDIATE_SIZE, device=self.device, dtype=dtype) if add_bias else None
        grad = torch.randn(*SHAPE[:-1], INTERMEDIATE_SIZE, device=self.device, dtype=dtype)

        expected = self._run_linear_func(x, weight, bias, grad, [])
        quack = self._run_linear_func(x, weight, bias, grad, [Kernel.quack_gemm])
        reference = self._run_linear_func(
            x.float(),
            weight.float(),
            bias.float() if bias is not None else None,
            grad.float(),
            [],
        )

        self.assert_close(quack.y, expected.y, dtype, reference.y)
        self.assert_close(quack.dx, expected.dx, dtype, reference.dx)
        self.assert_close(quack.dw, expected.dw, dtype, reference.dw)

        if add_bias:
            assert quack.db is not None
            assert expected.db is not None
            assert reference.db is not None
            self.assert_close(quack.db, expected.db, dtype, reference.db)

    def test_parameterized_linear_equivalence(self) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        Accelerator.set_seed(SEED)

        dtype = torch.float16
        x = torch.randn(*SHAPE, device=self.device, dtype=dtype)
        grad = torch.randn(*SHAPE[:-1], INTERMEDIATE_SIZE, device=self.device, dtype=dtype)

        module = ParameterizedLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            bias=True,
            std=0.02,
        ).to(device=self.device, dtype=dtype)

        reference_module = ParameterizedLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            bias=True,
            std=0.02,
        ).to(device=self.device, dtype=torch.float32)
        reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

        expected = self._run_linear_module(module, x, grad, [])
        quack = self._run_linear_module(module, x, grad, [Kernel.quack_gemm])
        reference = self._run_linear_module(reference_module, x.float(), grad.float(), [])

        self.assert_close(quack.y, expected.y, dtype, reference.y)
        self.assert_close(quack.dx, expected.dx, dtype, reference.dx)
        self.assert_close(quack.dw, expected.dw, dtype, reference.dw)
        assert quack.db is not None
        assert expected.db is not None
        assert reference.db is not None
        self.assert_close(quack.db, expected.db, dtype, reference.db)

    @parameterized.named_parameters(*_PLAIN_MLP_EQUIVALENCE_CASES)
    def test_plain_quack_gemm_mlp_equivalence(
        self,
        activation_function: str,
        use_interleaved_weights: bool,
        dtype: torch.dtype,
    ) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        Accelerator.set_seed(SEED)

        x = torch.randn(*SHAPE, device=self.device, dtype=dtype)
        grad = torch.randn(*SHAPE, device=self.device, dtype=dtype)

        module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=True,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=dtype)

        reference_module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=True,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=torch.float32)
        reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

        expected = self._run_mlp(module, x, grad, [])
        quack = self._run_mlp(module, x, grad, [Kernel.quack_gemm])
        reference = self._run_mlp(reference_module, x.float(), grad.float(), [])

        self.assert_close(quack.y, expected.y, dtype, reference.y)
        self.assert_close(quack.dx, expected.dx, dtype, reference.dx)
        self.assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
        self.assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)
        assert quack.db_fc is not None
        assert quack.db_proj is not None
        assert expected.db_fc is not None
        assert expected.db_proj is not None
        assert reference.db_fc is not None
        assert reference.db_proj is not None
        self.assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
        self.assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)

    @parameterized.named_parameters(*_COMBINED_MLP_EQUIVALENCE_CASES)
    def test_combined_quack_gemm_mlp_equivalence(
        self,
        kernels: list[Kernel],
        activation_function: str,
        use_interleaved_weights: bool,
    ) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        Accelerator.set_seed(SEED)

        dtype = torch.float16
        x = torch.randn(*SHAPE, device=self.device, dtype=dtype)
        grad = torch.randn(*SHAPE, device=self.device, dtype=dtype)

        module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=True,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=dtype)

        reference_module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=True,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=torch.float32)
        reference_module.load_state_dict({name: tensor.float() for name, tensor in module.state_dict().items()})

        expected = self._run_mlp(module, x, grad, [])
        quack = self._run_mlp(module, x, grad, kernels)
        reference = self._run_mlp(reference_module, x.float(), grad.float(), [])

        self.assert_close(quack.y, expected.y, dtype, reference.y)
        self.assert_close(quack.dx, expected.dx, dtype, reference.dx)
        self.assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
        self.assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)
        assert quack.db_fc is not None
        assert quack.db_proj is not None
        assert expected.db_fc is not None
        assert expected.db_proj is not None
        assert reference.db_fc is not None
        assert reference.db_proj is not None
        self.assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
        self.assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)

    @parameterized.named_parameters(*_EQUIVALENCE_CASES)
    def test_equivalence(
        self,
        kernel: Kernel,
        activation_function: str,
        use_interleaved_weights: bool,
        add_bias: bool,
        dtype: torch.dtype,
    ) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        Accelerator.set_seed(SEED)

        x = torch.randn(*SHAPE, device=self.device, dtype=dtype)
        grad = torch.randn(*SHAPE, device=self.device, dtype=dtype)

        self.mlp_module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=add_bias,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=dtype)

        self.reference_mlp_module = self.mlp_fn(
            activation_function=activation_function,
            add_bias=add_bias,
            use_interleaved_weights=use_interleaved_weights,
        ).to(device=self.device, dtype=torch.float32)
        self.reference_mlp_module.load_state_dict(
            {name: tensor.float() for name, tensor in self.mlp_module.state_dict().items()}
        )

        expected = self._run_mlp(self.mlp_module, x, grad, [])
        quack = self._run_mlp(self.mlp_module, x, grad, [kernel])
        reference = self._run_mlp(self.reference_mlp_module, x.float(), grad.float(), [])

        self.assert_close(quack.y, expected.y, dtype, reference.y)
        self.assert_close(quack.dx, expected.dx, dtype, reference.dx)
        self.assert_close(quack.dc_fc, expected.dc_fc, dtype, reference.dc_fc)
        self.assert_close(quack.dc_proj, expected.dc_proj, dtype, reference.dc_proj)

        if add_bias:
            assert quack.db_fc is not None
            assert quack.db_proj is not None
            assert expected.db_fc is not None
            assert expected.db_proj is not None
            assert reference.db_fc is not None
            assert reference.db_proj is not None
            self.assert_close(quack.db_fc, expected.db_fc, dtype, reference.db_fc)
            self.assert_close(quack.db_proj, expected.db_proj, dtype, reference.db_proj)

    def test_non_interleaved_glu(self) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        self.mlp_module = self.mlp_fn(
            activation_function="swiglu",
            add_bias=False,
            use_interleaved_weights=False,
        ).to(device=self.device, dtype=torch.float32)

        with enable_kernels([Kernel.quack_gemm_gated]):
            with pytest.raises(ValueError, match="requires use_interleaved_weights=True"):
                self.mlp_module(torch.randn(*SHAPE, device=self.device))

    def test_unsupported_gemm_act_activation(self) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        self.mlp_module = self.mlp_fn(
            activation_function="gelu",
            add_bias=False,
            use_interleaved_weights=False,
        ).to(device=self.device, dtype=torch.float32)

        with enable_kernels([Kernel.quack_gemm_act]):
            with pytest.raises(ValueError, match="is not supported by quack_gemm_act"):
                self.mlp_module(torch.randn(*SHAPE, device=self.device))

    def test_unsupported_gemm_gated_activation(self) -> None:
        skip_test_if_device_unavailable(self.device)

        if not is_quack_available():
            pytest.skip("skipping test because quack-kernels is unavailable")

        self.mlp_module = self.mlp_fn(
            activation_function="geglu",
            add_bias=False,
            use_interleaved_weights=True,
        ).to(device=self.device, dtype=torch.float32)

        with enable_kernels([Kernel.quack_gemm_gated]):
            with pytest.raises(ValueError, match="is not supported by quack_gemm_gated"):
                self.mlp_module(torch.randn(*SHAPE, device=self.device))

    def test_quack_gemm_gated_excludes_swiglu_packed(self) -> None:
        with pytest.raises(ValueError, match="quack_gemm_gated cannot be enabled with swiglu_packed"):
            KernelArgs(kernels=[Kernel.quack_gemm_gated, Kernel.swiglu_packed])

    def test_quack_gemm_act_allows_swiglu_packed(self) -> None:
        KernelArgs(kernels=[Kernel.quack_gemm_act, Kernel.swiglu_packed])

    def test_quack_gemm_allows_gemm_act(self) -> None:
        KernelArgs(kernels=[Kernel.quack_gemm, Kernel.quack_gemm_act])

    def test_quack_gemm_allows_gemm_gated(self) -> None:
        KernelArgs(kernels=[Kernel.quack_gemm, Kernel.quack_gemm_gated])

    def test_quack_gemm_act_and_gated_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="quack_gemm_act cannot be enabled with quack_gemm_gated"):
            KernelArgs(kernels=[Kernel.quack_gemm_act, Kernel.quack_gemm_gated])


if __name__ == "__main__":
    absltest.main()
