# **************************************************
# Copyright (c) 2026, Jyo Pari, Mayank Mishra
# **************************************************

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch
from einops import reduce, repeat

from .....utils import is_fla_available


def maybe_broadcast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor | None,
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
    assert num_heads % num_q_heads == 0
    assert num_heads % num_k_heads == 0
    assert num_heads % num_v_heads == 0
    assert num_heads % num_b_heads == 0

    q = repeat(q, "... h d -> ... (h g) d", g=num_heads // num_q_heads).contiguous()
    k = repeat(k, "... h d -> ... (h g) d", g=num_heads // num_k_heads).contiguous()
    v = repeat(v, "... h d -> ... (h g) d", g=num_heads // num_v_heads).contiguous()
    b = repeat(b, "... h   -> ... (h g)  ", g=num_heads // num_b_heads).contiguous()

    if initial_state is not None:
        assert initial_state.ndim == 4
        initial_state = repeat(
            initial_state, "b ... -> (b g) ...", g=q.shape[0] // initial_state.shape[0]
        ).contiguous()

    return q, k, v, b, initial_state


if is_fla_available():
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.delta_rule.chunk import chunk_delta_rule_bwd, chunk_delta_rule_fwd
    from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

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
            use_q_l2norm_in_kernel: bool = False,
            use_k_l2norm_in_kernel: bool = False,
            cu_seqlens: torch.LongTensor | None = None,
        ):
            if use_q_l2norm_in_kernel:
                q, q_rstd = l2norm_fwd(q)
            else:
                q_rstd = None

            if use_k_l2norm_in_kernel:
                k, k_rstd = l2norm_fwd(k)
            else:
                k_rstd = None

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
            )
            o, A, final_state = chunk_delta_rule_fwd(
                q=q_broadcast,
                k=k_broadcast,
                v=v_broadcast,
                beta=beta_broadcast,
                scale=scale,
                initial_state=initial_state_broadcast,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state)
            ctx.scale = scale
            ctx.cu_seqlens = cu_seqlens
            ctx.use_q_l2norm_in_kernel = use_q_l2norm_in_kernel
            ctx.use_k_l2norm_in_kernel = use_k_l2norm_in_kernel
            return o.to(q.dtype), final_state

        @staticmethod
        @input_guard
        @autocast_custom_bwd
        def backward(
            ctx,
            do: torch.Tensor,
            dht: torch.Tensor,
        ):
            q, q_rstd, k, k_rstd, v, beta, A, initial_state = ctx.saved_tensors

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
            )
            dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
                q=q_broadcast,
                k=k_broadcast,
                v=v_broadcast,
                beta=beta_broadcast,
                A=A,
                scale=ctx.scale,
                initial_state=initial_state_broadcast,
                do=do,
                dht=dht,
                cu_seqlens=ctx.cu_seqlens,
            )

            dq = reduce(dq, "... (h g) d -> ... h d", "sum", g=q_broadcast.shape[2] // q.shape[2])
            dk = reduce(dk, "... (h g) d -> ... h d", "sum", g=k_broadcast.shape[2] // k.shape[2])
            dv = reduce(dv, "... (h g) d -> ... h d", "sum", g=v_broadcast.shape[2] // v.shape[2])
            db = reduce(db, "... (h g)   -> ... h  ", "sum", g=beta_broadcast.shape[2] // beta.shape[2])

            if dh0 is not None:
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
                dq.to(q.dtype),
                dk.to(k.dtype),
                dv.to(v.dtype),
                db.to(beta.dtype),
                None,
                dh0,
                None,
                None,
                None,
                None,
                None,
            )

    @torch.compiler.disable
    def chunk_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_q_l2norm_in_kernel: bool = False,
        use_k_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        head_first: bool = False,
        allow_fp32: bool = False,
    ):
        r"""
        Args:
            q (torch.Tensor):
                queries of shape `[B, T, H, K]`.
            k (torch.Tensor):
                keys of shape `[B, T, H, K]`.
            v (torch.Tensor):
                values of shape `[B, T, H, V]`.
            beta (torch.Tensor):
                betas of shape `[B, T, H]`.
            scale (Optional[float]):
                Scale factor for the RetNet attention scores.
                If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
            initial_state (Optional[torch.Tensor]):
                Initial state of shape `[N, H, K, V]` for `N` input sequences.
                For equal-length input sequences, `N` equals the batch size `B`.
                Default: `None`.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
            use_qk_l2norm_in_kernel (Optional[bool]):
                Whether to use qk l2norm within the kernel for saving GPU memory.
                Default: `False`.
            cu_seqlens (torch.LongTensor):
                Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
                consistent with the FlashAttention API.
            head_first (Optional[bool]):
                Whether the inputs are in the head-first format. Default: `False`.
                This argument has been deprecated.

        Returns:
            o (torch.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

        Examples::
            >>> import torch
            >>> import torch.nn.functional as F
            >>> from einops import rearrange
            >>> from fla.ops.delta_rule import chunk_delta_rule
            # inputs with equal lengths
            >>> B, T, H, K, V = 4, 2048, 4, 512, 512
            >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
            >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
            >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
            >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
            >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
            >>> o, ht = chunk_delta_rule(
                q, k, v, beta,
                initial_state=h0,
                output_final_state=True
            )
            # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
            >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
            # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
            >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
            >>> o, ht = chunk_delta_rule(
                q, k, v, beta,
                initial_state=h0,
                output_final_state=True,
                cu_seqlens=cu_seqlens
            )
        """
        assert q.dtype == k.dtype == v.dtype
        assert (
            q.dtype != torch.float32 or allow_fp32
        ), "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
        assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

        if head_first:
            raise DeprecationWarning(
                "head_first is deprecated and will be removed in a future version. "
                "Please use head_first=False for now instead.",
            )
        if not head_first and q.shape[1] < q.shape[2]:
            warnings.warn(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "when head_first=False was specified. "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
            )
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
        scale = k.shape[-1] ** -0.5 if scale is None else scale
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
        )
        return o, final_state
