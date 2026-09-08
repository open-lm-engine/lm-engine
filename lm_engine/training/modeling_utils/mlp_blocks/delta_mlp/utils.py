# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
from einops import rearrange, reduce, repeat

from .....utils import is_fla_available


def maybe_broadcast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
    broadcast_value: bool,
    broadcast_initial_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert q.ndim == 4
    assert k.ndim == 4
    assert v.ndim == 4
    assert b.ndim == 3

    num_q_heads = q.shape[2]
    num_k_heads = k.shape[2]
    num_v_heads = v.shape[2]
    num_b_heads = b.shape[2]
    num_heads = max(num_q_heads, num_k_heads, num_v_heads, num_b_heads)
    assert num_heads == num_q_heads
    assert num_heads == num_k_heads
    assert num_heads == num_b_heads
    assert num_heads % num_v_heads == 0

    if broadcast_value:
        v = repeat(v, "... h d -> ... (h g) d", g=num_heads // num_v_heads).contiguous()

    if initial_state is not None:
        assert initial_state.ndim == 4
        if broadcast_initial_state:
            initial_state = repeat(
                initial_state,
                "b ... -> (b g) ...",
                g=q.shape[0] // initial_state.shape[0],
            ).contiguous()

    return q, k, v, b, initial_state


if is_fla_available():
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.cp import FLACPContext
    from fla.ops.delta_rule.chunk import prepare_chunk_indices
    from fla.ops.delta_rule.fused_recurrent import fused_recurrent_delta_rule_fwd
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
    from xma.functional.delta_rule.chunk import chunk_delta_rule_bwd, chunk_delta_rule_fwd

    class ChunkDeltaRuleFunction(torch.autograd.Function):

        @staticmethod
        @input_guard
        @autocast_custom_fwd
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            use_q_l2norm_in_kernel: bool,
            use_k_l2norm_in_kernel: bool,
            cu_seqlens: torch.LongTensor | None,
            cu_seqlens_cpu: torch.LongTensor | None,
            cp_context: FLACPContext | None,
            transpose_state_layout: bool,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:

            if use_q_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
            else:
                q_rstd = None

            if use_k_l2norm_in_kernel:
                k, k_rstd = l2norm_fwd(k)
            else:
                k_rstd = None

            chunk_indices = (
                prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu)
                if cu_seqlens is not None
                else None
            )
            (
                q_broadcast,
                k_broadcast,
                v_broadcast,
                beta_broadcast,
                initial_state_broadcast,
            ) = maybe_broadcast(
                q=q,
                k=k,
                v=v,
                b=beta,
                initial_state=initial_state,
                broadcast_value=False,
                broadcast_initial_state=cu_seqlens is None,
            )
            o, A, final_state, initial_state_broadcast_cp = chunk_delta_rule_fwd(
                q=q_broadcast,
                k=k_broadcast,
                v=v_broadcast,
                beta=beta_broadcast,
                scale=scale,
                initial_state=initial_state_broadcast,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=transpose_state_layout,
                cp_pipeline=False,
                compute_in_fp32=False,
            )
            o = reduce(o, "b t h d -> b t d", "sum")
            ctx.save_for_backward(
                q,
                q_rstd,
                k,
                k_rstd,
                v,
                beta,
                A,
                initial_state_broadcast_cp if cp_context is not None else initial_state,
                cu_seqlens,
                chunk_indices,
            )
            ctx.scale = scale
            ctx.use_q_l2norm_in_kernel = use_q_l2norm_in_kernel
            ctx.use_k_l2norm_in_kernel = use_k_l2norm_in_kernel
            ctx.cp_context = cp_context
            ctx.transpose_state_layout = transpose_state_layout
            ctx.initial_state_was_none = initial_state is None
            return o.to(q.dtype), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            torch.Tensor | None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]:

            q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors

            (
                q_broadcast,
                k_broadcast,
                v_broadcast,
                beta_broadcast,
                initial_state_broadcast,
            ) = maybe_broadcast(
                q=q,
                k=k,
                v=v,
                b=beta,
                initial_state=initial_state,
                broadcast_value=False,
                broadcast_initial_state=cu_seqlens is None,
            )
            # we reduces dv to v's original num heads inside `chunk_delta_rule_bwd`
            dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
                q=q_broadcast,
                k=k_broadcast,
                v=v_broadcast,
                beta=beta_broadcast,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state_broadcast,
                # we reduce `o` across heads in the forward pass, this becomes
                # broadcast across heads in the backward pass. `chunk_delta_rule_bwd`
                # takes care of the broadcasting inside the kernel.
                do=rearrange(do, "b t d -> b t 1 d"),
                dht=dht,
                cu_seqlens=cu_seqlens,
                cp_context=ctx.cp_context,
                chunk_indices=chunk_indices,
                transpose_state_layout=ctx.transpose_state_layout,
                cp_pipeline=False,
                compute_in_fp32=False,
            )

            if ctx.initial_state_was_none:
                # in CP setting, `dh0` is always a Tensor, so we need to
                # use `ctx.initial_state_was_none` to decide if we need `dh0`
                dh0 = None
            else:
                assert dh0 is not None
                assert initial_state is not None
                assert initial_state_broadcast is not None
                dh0 = reduce(
                    dh0, "(b g) ... -> b ...  ", "sum", g=initial_state_broadcast.shape[0] // initial_state.shape[0]
                )

            if ctx.use_q_l2norm_in_kernel:
                dq = l2norm_bwd(q, q_rstd, dq)
            if ctx.use_k_l2norm_in_kernel:
                dk = l2norm_bwd(k, k_rstd, dk)

            return (
                dq.to(q.dtype),  # q
                dk.to(k.dtype),  # k
                dv.to(v.dtype),  # v
                db.to(beta.dtype),  # beta
                None,  # scale
                dh0,  # initial_state
                None,  # output_final_state
                None,  # use_q_l2norm_in_kernel
                None,  # use_k_l2norm_in_kernel
                None,  # cu_seqlens
                None,  # cu_seqlens_cpu
                None,  # cp_context
                None,  # transpose_state_layout
            )

    @torch.compiler.disable
    def chunk_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_q_l2norm_in_kernel: bool = False,
        use_k_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        cp_context: FLACPContext | None = None,
        transpose_state_layout: bool = False,
    ):
        # Validate head dimensions
        if q.shape[2] != k.shape[2]:
            raise ValueError(
                f"q and k must have the same number of heads, "
                f"but got q.shape[2]={q.shape[2]} and k.shape[2]={k.shape[2]}"
            )

        if cp_context is not None:
            assert output_final_state is False, "Output final state is not supported for CP"
            assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
            cu_seqlens = cp_context.cu_seqlens
            if cp_context.cu_seqlens_cpu is not None:
                cu_seqlens_cpu = cp_context.cu_seqlens_cpu

        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing.",
                )
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
                )

        if scale is None:
            scale = k.shape[-1] ** -0.5

        o, final_state = ChunkDeltaRuleFunction.apply(
            q,
            k,
            v,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_q_l2norm_in_kernel,
            use_k_l2norm_in_kernel,
            cu_seqlens,
            cu_seqlens_cpu,
            cp_context,
            transpose_state_layout,
        )
        return o, final_state

    class FusedRecurrentFunction(torch.autograd.Function):

        @staticmethod
        @input_guard
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            use_q_l2norm_in_kernel: bool,
            use_k_l2norm_in_kernel: bool,
            cu_seqlens: torch.LongTensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:

            if use_q_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
            else:
                pass

            if use_k_l2norm_in_kernel:
                k, k_rstd = l2norm_fwd(k)
            else:
                pass

            (
                q_broadcast,
                k_broadcast,
                v_broadcast,
                beta_broadcast,
                initial_state_broadcast,
            ) = maybe_broadcast(
                q=q,
                k=k,
                v=v,
                b=beta,
                initial_state=initial_state,
                broadcast_value=True,
                broadcast_initial_state=cu_seqlens is None,
            )
            o, u, final_state = fused_recurrent_delta_rule_fwd(
                q=q_broadcast,
                k=k_broadcast,
                v=v_broadcast,
                beta=beta_broadcast,
                scale=scale,
                initial_state=initial_state_broadcast,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            o = reduce(o, "b t h d -> b t d", "sum")
            return o, final_state

        @staticmethod
        @input_guard
        def backward(ctx, do, dht):
            raise NotImplementedError

    def fused_recurrent_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        use_q_l2norm_in_kernel: bool = False,
        use_k_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing.",
                )
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
                )
        if scale is None:
            scale = k.shape[-1] ** -0.5

        o, final_state = FusedRecurrentFunction.apply(
            q,
            k,
            v,
            beta,
            scale,
            initial_state,
            output_final_state,
            use_q_l2norm_in_kernel,
            use_k_l2norm_in_kernel,
            cu_seqlens,
        )
        return o, final_state
