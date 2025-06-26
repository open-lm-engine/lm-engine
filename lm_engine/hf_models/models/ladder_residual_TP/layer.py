# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.distributed._functional_collectives as funcol
import torch.nn.functional as F

from ....dtensors import dtensor_to_tensor
from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager, is_cute_kernels_available
from ...cache import GenerationCache
from ...mixins import Block_TP
from ...modeling_utils_TP import get_mlp_block_TP
from ..ladder_residual.layer import LadderResidualBlock


if is_cute_kernels_available():
    from cute_kernels.constants import MAX_TRITON_BLOCK_SIZE
    from cute_kernels.math import ceil_divide, divide_if_divisible, get_next_power_of_2
    from cute_kernels.ops.rmsnorm import rmsnorm_backward_triton, rmsnorm_forward_triton
    from cute_kernels.ops.swiglu import swiglu_backward_triton, swiglu_forward_triton
    from cute_kernels.utils import ensure_contiguous, get_num_elements_and_hidden_size, get_sm_count

    @ensure_contiguous
    def _swiglu_packed_forward(x: torch.Tensor) -> torch.Tensor:
        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 64
        BLOCK_SIZE_H = 64

        output = torch.empty(*x.size()[:-1], divide_if_divisible(H, 2), device=x.device, dtype=x.dtype)

        with torch.cuda.device(x.device):
            swiglu_forward_triton[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
                x_ptr=x,
                output_ptr=output,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return output

    @ensure_contiguous
    def _swiglu_packed_backward(x: torch.Tensor, output_grad: torch.Tensor) -> torch.Tensor:
        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 64
        BLOCK_SIZE_H = 64

        x_grad = torch.empty_like(x)

        with torch.cuda.device(x.device):
            swiglu_backward_triton[ceil_divide(B, BLOCK_SIZE_B), ceil_divide(H, BLOCK_SIZE_H)](
                x_ptr=x,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return x_grad

    @ensure_contiguous
    def _rmsnorm_forward(
        x: torch.Tensor, weight: torch.Tensor | None = None, eps: float | None = None
    ) -> tuple[torch.Tensor]:
        if eps is None:
            eps = torch.finfo(x.dtype).eps

        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        output = torch.empty_like(x)
        rmsnorm_denominator = torch.empty(B, device=x.device, dtype=torch.float32)

        with torch.cuda.device(x.device):
            rmsnorm_forward_triton[ceil_divide(B, BLOCK_SIZE_B),](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_ptr=output,
                eps=eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        return output, rmsnorm_denominator

    @ensure_contiguous
    def _rmsnorm_backward(
        x: torch.Tensor,
        weight: torch.Tensor | None,
        rmsnorm_denominator: torch.Tensor,
        output_grad: torch.Tensor,
        eps: float | None = None,
    ) -> tuple[torch.Tensor | None]:
        B, H = get_num_elements_and_hidden_size(x)
        BLOCK_SIZE_B = 1
        BLOCK_SIZE_H = get_next_power_of_2(H)
        assert BLOCK_SIZE_H <= MAX_TRITON_BLOCK_SIZE

        x_grad = torch.empty_like(x)
        weight_grad = None if weight is None else torch.zeros_like(weight, dtype=torch.float32)

        sm_count = get_sm_count(x.device)
        num_programs = min(sm_count, ceil_divide(B, BLOCK_SIZE_B))

        with torch.cuda.device(x.device):
            rmsnorm_backward_triton[num_programs,](
                x_ptr=x,
                has_weight=weight is not None,
                weight_ptr=weight,
                output_grad_ptr=output_grad,
                x_grad_ptr=x_grad,
                weight_grad_ptr=weight_grad,
                eps=eps,
                has_rmsnorm_denominator=rmsnorm_denominator is not None,
                rmsnorm_denominator_ptr=rmsnorm_denominator,
                B=B,
                H=H,
                BLOCK_SIZE_B=BLOCK_SIZE_B,
                BLOCK_SIZE_H=BLOCK_SIZE_H,
            )

        if weight_grad is not None:
            weight_grad = weight_grad.type_as(weight)

        return x_grad, weight_grad

    def _mlp_forward(
        c_fc_input: torch.Tensor, c_fc_weight: torch.Tensor, c_proj_weight: torch.Tensor
    ) -> tuple[torch.Tensor]:
        swiglu_input = F.linear(c_fc_input, c_fc_weight)
        c_proj_input = _swiglu_packed_forward(swiglu_input)
        c_proj_output = F.linear(c_proj_input, c_proj_weight)
        return swiglu_input, c_proj_input, c_proj_output

    def _mlp_backward(
        c_fc_input: torch.Tensor,
        c_fc_weight: torch.Tensor,
        swiglu_input: torch.Tensor,
        c_proj_input: torch.Tensor,
        c_proj_weight: torch.Tensor,
        output_grad: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        c_proj_input_grad = F.linear(output_grad, c_proj_weight.T)
        c_fc_output_grad = _swiglu_packed_backward(swiglu_input, c_proj_input_grad)
        c_fc_input_grad = F.linear(c_fc_output_grad, c_fc_weight.T)
        c_fc_input_grad = funcol.all_reduce(
            c_fc_input_grad, reduceOp="sum", group=ProcessGroupManager.get_tensor_parallel_mesh()
        )

        c_proj_weight_grad = output_grad.transpose(-1, -2) @ c_proj_input.unsqueeze(0).unsqueeze(0)
        c_proj_weight_grad = c_proj_weight_grad.squeeze(0).squeeze(0)
        c_fc_weight_grad = c_fc_output_grad.transpose(-1, -2) @ c_fc_input.unsqueeze(0).unsqueeze(0)
        c_fc_weight_grad = c_fc_weight_grad.squeeze(0).squeeze(0)

        return c_fc_input_grad, c_fc_weight_grad, c_proj_weight_grad

    class _OverlappableBlock(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            current_attention_out: torch.Tensor | None,
            current_mlp_out: torch.Tensor | None,
            residual: torch.Tensor,
            ln_1_weight: torch.Tensor,
            ln_2_weight: torch.Tensor,
            attention_c_fc_weight: torch.Tensor,
            attention_c_proj_weight: torch.Tensor,
            mlp_c_fc_weight: torch.Tensor,
            mlp_c_proj_weight: torch.Tensor,
            eps1: float,
            eps2: float,
        ) -> tuple[torch.Tensor]:
            ctx.current_attention_out_is_none = current_attention_out is None
            ctx.current_mlp_out_is_none = current_mlp_out is None
            ctx.eps1 = eps1
            ctx.eps2 = eps2

            if not ctx.current_mlp_out_is_none:
                current_mlp_out = funcol.all_reduce(
                    current_mlp_out, reduceOp="sum", group=ProcessGroupManager.get_tensor_parallel_mesh()
                )

            if ctx.current_attention_out_is_none:
                attention_rmsnorm_input = (
                    residual.wait() if isinstance(residual, funcol.AsyncCollectiveTensor) else residual
                )
            else:
                attention_rmsnorm_input = residual + current_attention_out

            attention_input, attention_rmsnorm_denominator = _rmsnorm_forward(
                x=attention_rmsnorm_input, weight=ln_1_weight, eps=eps1
            )

            attention_swiglu_input, attention_c_proj_input, attention_c_proj_output = _mlp_forward(
                c_fc_input=attention_input, c_fc_weight=attention_c_fc_weight, c_proj_weight=attention_c_proj_weight
            )

            attention_c_proj_output = funcol.all_reduce(
                attention_c_proj_output, reduceOp="sum", group=ProcessGroupManager.get_tensor_parallel_mesh()
            )

            if ctx.current_mlp_out_is_none:
                mlp_rmsnorm_input = attention_rmsnorm_input
            else:
                mlp_rmsnorm_input = attention_rmsnorm_input + current_mlp_out

            mlp_input, mlp_rmsnorm_denominator = _rmsnorm_forward(x=mlp_rmsnorm_input, weight=ln_2_weight, eps=eps2)

            mlp_swiglu_input, mlp_c_proj_input, mlp_c_proj_output = _mlp_forward(
                c_fc_input=mlp_input, c_fc_weight=mlp_c_fc_weight, c_proj_weight=mlp_c_proj_weight
            )

            ctx.save_for_backward(
                # attention RMSNorm
                attention_rmsnorm_input,
                ln_1_weight,
                attention_rmsnorm_denominator,
                # attention
                attention_input,
                attention_c_fc_weight,
                attention_swiglu_input,
                attention_c_proj_input,
                attention_c_proj_weight,
                # MLP RMSNorm
                mlp_rmsnorm_input,
                ln_2_weight,
                mlp_rmsnorm_denominator,
                # MLP
                mlp_input,
                mlp_c_fc_weight,
                mlp_swiglu_input,
                mlp_c_proj_input,
                mlp_c_proj_weight,
            )

            return attention_c_proj_output, mlp_c_proj_output, mlp_rmsnorm_input

        @staticmethod
        def backward(
            ctx,
            attention_c_proj_output_grad: torch.Tensor,
            mlp_c_proj_output_grad: torch.Tensor,
            mlp_rmsnorm_input_grad: torch.Tensor,
        ) -> tuple[torch.Tensor]:
            (  # attention RMSNorm
                attention_rmsnorm_input,
                ln_1_weight,
                attention_rmsnorm_denominator,
                # attention
                attention_input,
                attention_c_fc_weight,
                attention_swiglu_input,
                attention_c_proj_input,
                attention_c_proj_weight,
                # MLP RMSNorm
                mlp_rmsnorm_input,
                ln_2_weight,
                mlp_rmsnorm_denominator,
                # MLP
                mlp_input,
                mlp_c_fc_weight,
                mlp_swiglu_input,
                mlp_c_proj_input,
                mlp_c_proj_weight,
            ) = ctx.saved_tensors

            mlp_input_grad, mlp_c_fc_weight_grad, mlp_c_proj_weight_grad = _mlp_backward(
                c_fc_input=mlp_input,
                c_fc_weight=mlp_c_fc_weight,
                swiglu_input=mlp_swiglu_input,
                c_proj_input=mlp_c_proj_input,
                c_proj_weight=mlp_c_proj_weight,
                output_grad=mlp_c_proj_output_grad,
            )

            attention_input_grad, attention_c_fc_weight_grad, attention_c_proj_weight_grad = _mlp_backward(
                c_fc_input=attention_input,
                c_fc_weight=attention_c_fc_weight,
                swiglu_input=attention_swiglu_input,
                c_proj_input=attention_c_proj_input,
                c_proj_weight=attention_c_proj_weight,
                output_grad=attention_c_proj_output_grad,
            )

            tmp, ln_2_weight_grad = _rmsnorm_backward(
                x=mlp_rmsnorm_input,
                weight=ln_2_weight,
                rmsnorm_denominator=mlp_rmsnorm_denominator,
                output_grad=mlp_input_grad.wait(),
                eps=ctx.eps2,
            )
            mlp_rmsnorm_input_grad = mlp_rmsnorm_input_grad + tmp
            del tmp

            attention_rmsnorm_input_grad = mlp_rmsnorm_input_grad
            current_mlp_out_grad = None if ctx.current_mlp_out_is_none else mlp_rmsnorm_input_grad

            tmp, ln_1_weight_grad = _rmsnorm_backward(
                x=attention_rmsnorm_input,
                weight=ln_1_weight,
                rmsnorm_denominator=attention_rmsnorm_denominator,
                output_grad=attention_input_grad.wait(),
                eps=ctx.eps1,
            )
            attention_rmsnorm_input_grad = attention_rmsnorm_input_grad + tmp
            del tmp

            residual_grad = attention_rmsnorm_input_grad
            current_attention_out_grad = None if ctx.current_attention_out_is_none else attention_rmsnorm_input_grad

            return (
                current_attention_out_grad,
                current_mlp_out_grad,
                residual_grad,
                ln_1_weight_grad,
                ln_2_weight_grad,
                attention_c_fc_weight_grad,
                attention_c_proj_weight_grad,
                mlp_c_fc_weight_grad,
                mlp_c_proj_weight_grad,
                None,  # eps1
                None,  # eps2
            )


class LadderResidualBlock_TP(Block_TP):
    def __init__(self, config, use_padding_free_transformer, layer_idx=None, sequence_parallel=False):
        super().__init__(config, use_padding_free_transformer, layer_idx, sequence_parallel)

        self.mlp0_block = get_mlp_block_TP(
            config,
            use_padding_free_transformer=use_padding_free_transformer,
            sequence_parallel=sequence_parallel,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        current_attention_out: torch.Tensor,
        current_mlp_out: torch.Tensor,
        residual: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor]:
        if is_kernel_allowed(Kernel.ladder_residual_overlapped_layer):
            assert self.m_residual in [None, 1]

            current_attention_out, current_mlp_out, residual = _OverlappableBlock.apply(
                current_attention_out,
                current_mlp_out,
                residual,
                dtensor_to_tensor(self.ln_1.weight),
                dtensor_to_tensor(self.ln_2.weight),
                dtensor_to_tensor(self.mlp0_block.c_fc.weight),
                dtensor_to_tensor(self.mlp0_block.c_proj.weight),
                dtensor_to_tensor(self.mlp_block.c_fc.weight),
                dtensor_to_tensor(self.mlp_block.c_proj.weight),
                self.ln_1.eps,
                self.ln_2.eps,
            )
        else:
            current_attention_out, current_mlp_out, residual = LadderResidualBlock.forward(
                self,
                current_attention_out=current_attention_out,
                current_mlp_out=current_mlp_out,
                residual=residual,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return current_attention_out, current_mlp_out, residual
