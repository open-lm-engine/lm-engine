# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from ....parallel import ProcessGroupManager
from ....utils import divide_if_divisible, is_mamba_2_ssm_available


if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def _pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (0, 0, 0, 0, 0, pad_size, 0, 0) if len(input_tensor.shape) == 4 else (0, 0, 0, pad_size, 0, 0)

    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def _reshape_into_chunks(input_tensor: torch.Tensor, pad_size: int, chunk_size: int) -> torch.Tensor:
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = _pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, num_heads] -> [bsz, -1, chunk_size, num_heads]
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        # [bsz, seq_len multiple of chunk_size, num_heads, head_dim or state_size] -> [bsz, -1, chunk_size, num_heads, head_dim or state_size]
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

    return input_tensor


def _segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def _all_gather_context_parallel_with_grad(input_tensor: torch.Tensor) -> torch.Tensor:
    cp_mesh = ProcessGroupManager.get_context_parallel_mesh()
    dtensor = DTensor.from_local(input_tensor.contiguous(), device_mesh=cp_mesh, placements=[Shard(0)])
    dtensor = dtensor.redistribute(placements=[Replicate()])

    return dtensor.to_local(grad_placements=[Partial()])


class _SerialPrefixScan(torch.autograd.Function):
    """Serial prefix scan over CP ranks with manual backward.

    Forward:  s[r] = exp_A[r] * s[r-1] + final[r],  s[-1] = 0
    Backward: chain-rule through the linear recurrence without re-entering autograd.
    """

    @staticmethod
    def forward(ctx, all_exp_A: torch.Tensor, all_final: torch.Tensor, cp_rank: int) -> torch.Tensor:
        # all_exp_A : [cp_world_size, batch, num_heads]
        # all_final : [cp_world_size, batch, num_heads, head_dim, state_size]
        prev_states = []
        s = torch.zeros_like(all_final[0])
        for r in range(cp_rank):
            prev_states.append(s)
            s = all_exp_A[r][:, :, None, None] * s + all_final[r]
        ctx.save_for_backward(all_exp_A, *prev_states)
        ctx.cp_rank = cp_rank
        ctx.all_final_shape = all_final.shape
        return s

    @staticmethod
    def backward(ctx, grad_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        all_exp_A = ctx.saved_tensors[0]
        prev_states = ctx.saved_tensors[1:]
        cp_rank = ctx.cp_rank

        grad_all_exp_A = torch.zeros_like(all_exp_A)
        grad_all_final = torch.zeros(ctx.all_final_shape, dtype=grad_s.dtype, device=grad_s.device)

        for r in range(cp_rank - 1, -1, -1):
            grad_all_final[r] = grad_s
            grad_all_exp_A[r] = (grad_s * prev_states[r]).sum(dim=(-2, -1))
            grad_s = grad_s * all_exp_A[r][:, :, None, None]

        return grad_all_exp_A, grad_all_final, None


def _serial_prefix_scan(all_exp_A: torch.Tensor, all_final: torch.Tensor, cp_rank: int) -> torch.Tensor:
    return _SerialPrefixScan.apply(all_exp_A, all_final, cp_rank)


def get_cp_initial_ssm_state(
    ssm_final_zero: torch.Tensor,
    dt: torch.Tensor,
    A_neg: torch.Tensor,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
) -> torch.Tensor:
    """Compute the correct initial SSM state for this CP rank.

    Uses all-gather + local prefix scan so every CP world size works exactly:
      s_init[r] = Phi[r-1] * s_init[r-1] + b[r-1]
    where Phi[r] = exp(A * Σ_t dt_t) is the chunk transition and b[r] is the
    zero-initial final state.
    """
    cp_rank = ProcessGroupManager.get_context_parallel_rank()
    cp_world_size = ProcessGroupManager.get_context_parallel_world_size()
    batch_size = ssm_final_zero.shape[0]

    # Diagonal transition factor: exp(A[h] * Σ_t dt_eff[b,t,h])
    exp_A_chunk = torch.exp(A_neg[None, :] * dt.float().sum(dim=1))  # (batch, num_heads)

    # All-gather both tensors from every rank (gathered along batch dim 0).
    all_exp_A = _all_gather_context_parallel_with_grad(exp_A_chunk)
    all_final = _all_gather_context_parallel_with_grad(ssm_final_zero)

    all_exp_A = all_exp_A.reshape(cp_world_size, batch_size, num_heads)
    all_final = all_final.reshape(cp_world_size, batch_size, num_heads, head_dim, ssm_state_size)

    return _serial_prefix_scan(all_exp_A, all_final, cp_rank)


def _mamba2_recurrent_step_torch(
    x: torch.Tensor,
    A_neg: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    ssm_state: torch.Tensor,
    num_heads: int,
    n_groups: int,
    head_dim: int,
    ssm_state_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token recurrent SSM update: pure torch fallback for `selective_state_update`.

    x, B, C, dt are for a single time step (already conv'd, and dt already
    softplus'd with bias applied). A is already `-exp(A_log)`. Returns `(y, new_ssm_state)`
    where `y` has shape (batch_size, 1, num_heads * head_dim) and `new_ssm_state` has shape
    (batch_size, num_heads, head_dim, ssm_state_size).
    """
    batch_size = x.shape[0]
    # We need to guarantee that anything regarding the cache is on the same device
    cache_device = ssm_state.device

    dt = dt.squeeze(1)

    # Note: there is no need to pad parameter matrices here, as there is just one new token
    # for batched generation
    A_neg = A_neg[..., None, None].expand(num_heads, head_dim, ssm_state_size).to(dtype=torch.float32)
    # A -> (N, head_dim, ssm_state_size)
    dA = (torch.exp(dt[:, :, None, None] * A_neg)).to(device=cache_device)
    # dA -> (B, N, head_dim, ssm_state_size)

    # Discretize B
    # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
    # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
    # NOTE: S = 1 actually here
    B, C = [i.reshape(batch_size, n_groups, -1)[..., None, :] for i in (B, C)]
    # B, C -> (B, G, 1, ssm_state_size)
    B, C = [i.expand(batch_size, n_groups, num_heads // n_groups, i.shape[-1]).contiguous() for i in (B, C)]
    # B, C -> (B, G, N / G, ssm_state_size)
    B, C = [i.reshape(batch_size, -1, i.shape[-1]) for i in (B, C)]
    # B, C -> (B, N, ssm_state_size)

    # (B, N, 1, 1) * (B, N, 1, ssm_state_size), broadcasts against head_dim below
    # B is same as k and is shared across heads and dt is used to expand it
    dB = dt[..., None, None] * B[..., None, :]
    # dB -> (B, N, 1, ssm_state_size)

    # Discretize x into dB
    x = x.reshape(batch_size, -1, head_dim)
    # x -> (B, N, head_dim)
    dBx = (dB * x[..., None]).to(device=cache_device)
    # dBx -> (B, N, head_dim, ssm_state_size)

    # State calculation
    ssm_state = ssm_state * dA + dBx

    ssm_state_for_output = ssm_state.to(device=C.device, dtype=C.dtype)  # Shape: [b, h, d, n]
    # Reshape ssm_states to merge the first two dimensions
    ssm_states_reshaped = ssm_state_for_output.view(batch_size * num_heads, head_dim, ssm_state_size)  # [b*h, d, n]
    C_reshaped = C.view(batch_size * num_heads, ssm_state_size, 1)  # Shape: [b*h, n, 1]
    y = torch.bmm(ssm_states_reshaped, C_reshaped)
    y = y.view(batch_size, num_heads, head_dim)

    # D skip connection
    # [num_heads] -> [num_heads, head_dim]
    D = D[..., None].expand(D.shape[0], head_dim)
    y = (y + x * D).to(y.dtype)

    # [bsz, num_heads, head_dim] -> [bsz, 1, intermediate_size]
    y = y.reshape(batch_size, -1)[:, None, ...]

    return y, ssm_state


def _mamba2_chunk_scan_torch(
    x: torch.Tensor,
    A_neg: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    chunk_size: int,
    num_heads: int,
    n_groups: int,
    head_dim: int,
    ssm_state_size: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive chunked SSD scan: pure torch fallback for `mamba_chunk_scan_combined`.

    x, B, C, dt are for the full sequence (already conv'd, and dt already
    softplus'd with bias applied). A is already `-exp(A_log)`; `A_log` itself is only needed
    for the context-parallel initial-state correction. Returns `(y, final_ssm_state)` where
    `y` has shape (batch_size, seq_len, num_heads * head_dim) and `final_ssm_state` has shape
    (batch_size, num_heads, head_dim, ssm_state_size).
    """
    batch_size = x.shape[0]

    x = x.reshape(batch_size, seq_len, -1, head_dim).float()
    B = B.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    C = C.reshape(batch_size, seq_len, -1, ssm_state_size).float()
    B = B.repeat(1, 1, num_heads // n_groups, 1)
    C = C.repeat(1, 1, num_heads // n_groups, 1)
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    D_residual = D[..., None] * _pad_tensor_by_size(x, pad_size)

    # Discretize x and A
    x = x * dt[..., None]
    A = A_neg.to(x.dtype) * dt

    # Rearrange into blocks/chunks
    x, A, B, C = [_reshape_into_chunks(t, pad_size, chunk_size) for t in (x, A, B, C)]

    # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # This is the analog of a causal mask
    L = torch.exp(_segment_sum(A))

    # Contraction of C and B to get G (attention-weights like)
    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # shape: (b, c, l, s, h, n)
    G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)

    # Compute M, equivalent to applying attention mask to weights
    M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
    M = M_intermediate.sum(dim=-1)

    # Compute Y_diag (apply to values)
    Y_diag = (M[..., None] * x[:, :, None]).sum(dim=3)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
    states = (B_decay[..., None, :] * x[..., None]).sum(dim=2)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    decay_chunk = torch.exp(_segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    decay_chunk = decay_chunk.transpose(1, 3)

    if ProcessGroupManager.is_context_parallel_enabled():
        # Get final state with zero initial to compute the correct CP initial state
        states_zero = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
        new_states_zero = (decay_chunk[..., None, None] * states_zero[:, :, None, ...]).sum(dim=1)
        ssm_state_zero = new_states_zero[:, -1]
        previous_states = get_cp_initial_ssm_state(ssm_state_zero, dt, A_neg, num_heads, head_dim, ssm_state_size)
        previous_states = previous_states[:, None, ...].to(states.dtype)
    else:
        previous_states = torch.zeros_like(states[:, :1])

    states = torch.cat([previous_states, states], dim=1)
    new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
    states, ssm_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    C_times_states = C[..., None, :] * states[:, :, None, ...]
    state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
    Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    y = Y_diag + Y_off
    # [bsz, -1, chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
    y = y.reshape(batch_size, -1, num_heads, head_dim)

    y = y + D_residual
    # Cutting off padded chunks
    if pad_size > 0:
        y = y[:, :seq_len, :, :]
    y = y.reshape(batch_size, seq_len, -1)

    return y, ssm_state


def mamba2_torch(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_neg: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    h: torch.Tensor | None,
    use_recurrent: bool,
    intermediate_size: int,
    num_groups: int,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, S, _ = x.size()

    if use_recurrent:
        if h is None:
            h = torch.zeros(
                batch_size,
                num_heads,
                divide_if_divisible(intermediate_size, num_heads),
                ssm_state_size,
                device=x.device,
                dtype=x.dtype,
            )

        x, h = _mamba2_recurrent_step_torch(
            x=x,
            A_neg=A_neg,
            B=B,
            C=C,
            D=D,
            dt=dt,
            ssm_state=h,
            num_heads=num_heads,
            n_groups=num_groups,
            head_dim=head_dim,
            ssm_state_size=ssm_state_size,
        )
    else:
        assert h is None

        x, h = _mamba2_chunk_scan_torch(
            x=x,
            A_neg=A_neg,
            B=B,
            C=C,
            D=D,
            dt=dt,
            chunk_size=chunk_size,
            num_heads=num_heads,
            n_groups=num_groups,
            head_dim=head_dim,
            ssm_state_size=ssm_state_size,
            seq_len=S,
        )

    return x, h


def mamba2_cuda(
    x: torch.Tensor,
    A_neg: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    h: torch.Tensor | None,
    output_state: bool,
    num_groups: int,
    num_heads: int,
    head_dim: int,
    ssm_state_size: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, S, _ = x.size()

    if output_state:
        x = selective_state_update(
            h,
            x.view(batch_size, num_heads, head_dim),
            dt[:, :, None].expand(-1, -1, head_dim),
            A_neg[:, None, ...][:, :, None].expand(-1, head_dim, ssm_state_size).to(dtype=torch.float32),
            B.reshape(batch_size, num_groups, -1),
            C.reshape(batch_size, num_groups, -1),
            D[:, None, ...].expand(-1, head_dim),
            z=None,
            dt_softplus=False,
        )

        x = x.view(batch_size, num_heads * head_dim)[:, None, ...]
    else:
        assert h is None

        if ProcessGroupManager.is_context_parallel_enabled():
            scan_output_zero, ssm_state_zero = mamba_chunk_scan_combined(
                x.view(batch_size, S, -1, head_dim),
                dt,
                A_neg,
                B.view(batch_size, S, num_groups, -1),
                C.view(batch_size, S, num_groups, -1),
                chunk_size=chunk_size,
                D=D,
                z=None,
                seq_idx=None,
                return_final_states=True,
                dt_softplus=False,
            )

            ssm_state_zero = ssm_state_zero + scan_output_zero.sum().to(ssm_state_zero.dtype) * 0
            h = get_cp_initial_ssm_state(ssm_state_zero, dt, A_neg, num_heads, head_dim, ssm_state_size)

        x, h = mamba_chunk_scan_combined(
            x.view(batch_size, S, -1, head_dim),
            dt,
            A_neg,
            B.view(batch_size, S, num_groups, -1),
            C.view(batch_size, S, num_groups, -1),
            chunk_size=chunk_size,
            D=D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_softplus=False,
            initial_states=h,
        )

        x = x.view(batch_size, S, -1)

    return x, h
